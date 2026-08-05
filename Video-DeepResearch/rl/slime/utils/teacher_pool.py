from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
import yaml

_DEFAULT_GENERATE_PATH = "/generate"
_POOL_CACHE: dict[tuple[str, str], "TeacherPool"] = {}


@dataclass(slots=True)
class TeacherEndpoint:
    name: str
    url: str
    max_concurrency: int = 1
    inflight: int = 0
    success_count: int = 0
    failure_count: int = 0
    cooldown_until: float = 0.0
    last_error: str | None = None

    @property
    def healthy(self) -> bool:
        return time.time() >= self.cooldown_until


@dataclass(slots=True)
class TeacherPool:
    model_name: str
    endpoints: list[TeacherEndpoint]
    request_timeout_sec: float = 600.0
    max_retries: int = 2
    retry_backoff_sec: float = 0.5
    cooldown_sec: float = 30.0
    failure_threshold: int = 3
    _select_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @classmethod
    def from_config(cls, config_path: str, model_name: str) -> "TeacherPool":
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Teacher pool config not found: {config_path}")

        with config_file.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        models = config.get("models")
        if not isinstance(models, dict):
            raise ValueError("Teacher pool config must contain a top-level 'models' mapping.")

        model_cfg = models.get(model_name)
        if not isinstance(model_cfg, dict):
            raise ValueError(
                f"Teacher model '{model_name}' not found in teacher pool config {config_path}. "
                f"Available models: {sorted(models)}"
            )

        endpoints_cfg = model_cfg.get("endpoints")
        if not isinstance(endpoints_cfg, list) or not endpoints_cfg:
            raise ValueError(f"Teacher model '{model_name}' must define a non-empty endpoints list.")

        endpoints: list[TeacherEndpoint] = []
        for idx, item in enumerate(endpoints_cfg):
            if not isinstance(item, dict):
                raise ValueError(f"Teacher endpoint entry at index {idx} must be a mapping.")
            name = str(item.get("name") or f"{model_name}-{idx}")
            url = item.get("url")
            if url is None:
                host = item.get("host")
                port = item.get("port")
                if host is None or port is None:
                    raise ValueError(
                        f"Teacher endpoint '{name}' must define either 'url' or both 'host' and 'port'."
                    )
                path = str(item.get("path") or _DEFAULT_GENERATE_PATH)
                if not path.startswith("/"):
                    path = f"/{path}"
                url = f"http://{host}:{port}{path}"
            max_concurrency = int(item.get("max_concurrency", 1))
            if max_concurrency < 1:
                raise ValueError(f"Teacher endpoint '{name}' has invalid max_concurrency={max_concurrency}.")
            endpoints.append(TeacherEndpoint(name=name, url=str(url), max_concurrency=max_concurrency))

        return cls(
            model_name=model_name,
            endpoints=endpoints,
            request_timeout_sec=float(model_cfg.get("request_timeout_sec", 600.0)),
            max_retries=int(model_cfg.get("max_retries", 2)),
            retry_backoff_sec=float(model_cfg.get("retry_backoff_sec", 0.5)),
            cooldown_sec=float(model_cfg.get("cooldown_sec", 30.0)),
            failure_threshold=int(model_cfg.get("failure_threshold", 3)),
        )

    @classmethod
    def from_args(cls, args) -> "TeacherPool":
        config_path = getattr(args, "teacher_pool_config", None)
        model_name = getattr(args, "teacher_model_name", None)
        if not config_path or not model_name:
            raise ValueError("Teacher pool requires both args.teacher_pool_config and args.teacher_model_name.")

        cache_key = (str(config_path), str(model_name))
        pool = _POOL_CACHE.get(cache_key)
        if pool is None:
            pool = cls.from_config(str(config_path), str(model_name))
            _POOL_CACHE[cache_key] = pool
        return pool

    def get_inflight_stats(self) -> dict[str, Any]:
        return {
            "total_inflight": sum(endpoint.inflight for endpoint in self.endpoints),
            "total_capacity": sum(endpoint.max_concurrency for endpoint in self.endpoints),
            "per_endpoint": {
                endpoint.name: {
                    "inflight": endpoint.inflight,
                    "max_concurrency": endpoint.max_concurrency,
                    "healthy": endpoint.healthy,
                }
                for endpoint in self.endpoints
            },
        }

    async def request_json(self, payload: dict[str, Any], *, request_name: str = "teacher") -> tuple[dict[str, Any], TeacherEndpoint]:
        max_attempts = max(1, self.max_retries + 1)
        last_error: Exception | None = None
        excluded_names: set[str] = set()

        for attempt in range(max_attempts):
            endpoint = await self._acquire_endpoint(excluded_names)
            try:
                timeout = aiohttp.ClientTimeout(total=self.request_timeout_sec)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(endpoint.url, json=payload) as resp:
                        resp.raise_for_status()
                        result = await resp.json()
                await self._mark_success(endpoint)
                return result, endpoint
            except aiohttp.ClientResponseError as exc:
                await self._mark_failure(endpoint, exc)
                if 400 <= exc.status < 500:
                    raise
                last_error = exc
                excluded_names.add(endpoint.name)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                await self._mark_failure(endpoint, exc)
                last_error = exc
                excluded_names.add(endpoint.name)
            except Exception as exc:
                await self._mark_failure(endpoint, exc)
                raise

            if attempt + 1 < max_attempts:
                await asyncio.sleep(self.retry_backoff_sec)
                if len(excluded_names) >= len(self.endpoints):
                    excluded_names.clear()

        raise RuntimeError(
            f"{request_name} request failed for teacher model '{self.model_name}' after {max_attempts} attempts"
        ) from last_error

    async def _acquire_endpoint(self, excluded_names: set[str]) -> TeacherEndpoint:
        while True:
            async with self._select_lock:
                now = time.time()
                candidates = [
                    endpoint
                    for endpoint in self.endpoints
                    if endpoint.name not in excluded_names
                    and endpoint.healthy
                    and endpoint.inflight < endpoint.max_concurrency
                ]
                if candidates:
                    endpoint = min(candidates, key=lambda item: (item.inflight, item.name))
                    endpoint.inflight += 1
                    return endpoint

                recovering = [
                    endpoint for endpoint in self.endpoints if endpoint.name not in excluded_names and endpoint.cooldown_until > now
                ]
                sleep_for = min((endpoint.cooldown_until - now for endpoint in recovering), default=0.05)
            await asyncio.sleep(max(0.01, sleep_for))

    async def _mark_success(self, endpoint: TeacherEndpoint) -> None:
        async with self._select_lock:
            endpoint.inflight -= 1
            endpoint.success_count += 1
            endpoint.failure_count = 0
            endpoint.last_error = None
            endpoint.cooldown_until = 0.0

    async def _mark_failure(self, endpoint: TeacherEndpoint, exc: Exception) -> None:
        async with self._select_lock:
            endpoint.inflight -= 1
            endpoint.failure_count += 1
            endpoint.last_error = repr(exc)
            if endpoint.failure_count >= self.failure_threshold:
                endpoint.cooldown_until = time.time() + self.cooldown_sec


def get_teacher_pool(args) -> TeacherPool:
    return TeacherPool.from_args(args)

