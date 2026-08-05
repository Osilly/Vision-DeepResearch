"""
Vision-DeepResearch Tools - Shared utilities

Based on vision_deepresearch_async_workflow/tools/shared.py
Optimized for slime framework integration.
"""

import asyncio
import json
import os
import re
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def run_with_retries(func: Callable[[], T], attempts: int = 5, delay: float = 0.5) -> T:
    """Execute a callable with retry support."""
    last_error: Exception | None = None
    for attempt in range(1, max(attempts, 1) + 1):
        try:
            return func()
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                break
            if delay > 0:
                time.sleep(delay)
    if last_error is not None:
        raise last_error
    raise RuntimeError("run_with_retries executed without performing any attempts")


async def run_blocking(
    func: Callable[[], T], executor: ThreadPoolExecutor | None = None
) -> T:
    """Run a blocking call in the given executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, func)


async def run_with_retries_async(
    func: Callable[[], T],
    attempts: int = 5,
    delay: float = 1,
    executor: ThreadPoolExecutor | None = None,
) -> T:
    """Execute a callable with retry support without blocking the event loop."""
    last_error: Exception | None = None
    for attempt in range(1, max(attempts, 1) + 1):
        try:
            return await run_blocking(func, executor=executor)
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                break
            if delay > 0:
                await asyncio.sleep(delay)
    if last_error is not None:
        raise last_error
    raise RuntimeError("run_with_retries_async executed without performing any attempts")


def shorten_for_log(text: str, limit: int = 200) -> str:
    """Create a concise preview string for debug logging."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return ""
    normalized = text.replace("\n", "\\n")
    if len(normalized) <= limit * 2:
        return normalized
    return f"{normalized[:limit]} ... {normalized[-limit:]}"


def log_tool_event(
    source: str,
    status: str,
    message: str | None,
    *,
    error: str | None = None,
    level: str = "INFO",
) -> None:
    """Unified logging helper for DeepResearch tools."""
    safe_message = message or ""
    message_preview = shorten_for_log(safe_message)
    log_parts = [
        f"[Tool][{source}][{status}][{level}]",
        f"message_len={len(safe_message)}",
        f"preview={json.dumps(message_preview, ensure_ascii=False)}",
    ]
    if error is not None:
        error_preview = shorten_for_log(error)
        log_parts.append(f"error_len={len(error)}")
        log_parts.append(f"error={json.dumps(error_preview, ensure_ascii=False)}")
    print(" ".join(log_parts))


def log_search(
    source: str,
    status: str,
    query: str,
    result: str | None = None,
    error: str | None = None,
) -> None:
    """Standardized debug logs for search tools."""
    parts = [f"query={json.dumps(query, ensure_ascii=False)}"]
    if result is not None:
        preview = shorten_for_log(result)
        parts.append(f"result_len={len(result)}")
        parts.append(f"preview={json.dumps(preview, ensure_ascii=False)}")
    message = " ".join(parts)
    level = "ERROR" if error else "INFO"
    log_tool_event(
        source=f"Search/{source}",
        status=status,
        message=message,
        error=error,
        level=level,
    )


def parse_json_from_model_output(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from model text, tolerating Markdown fences."""
    if not isinstance(text, str) or not text.strip():
        return None

    candidate = text.strip()
    if candidate.startswith("```json"):
        candidate = candidate[7:].strip()
    elif candidate.startswith("```"):
        candidate = candidate[3:].strip()
    if candidate.endswith("```"):
        candidate = candidate[:-3].strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", candidate)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    if not isinstance(parsed, dict):
        return None

    for key in ("rational", "evidence", "summary"):
        parsed.setdefault(key, "")
    return parsed


def _extract_message_text_and_images(messages: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Flatten chat messages for SGLang /generate and collect image URLs/base64."""
    prompt_parts: list[str] = []
    image_data: list[str] = []

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        prompt_parts.append(f"<{role}>")

        if isinstance(content, str):
            prompt_parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    prompt_parts.append(str(item))
                    continue
                item_type = item.get("type")
                if item_type == "text":
                    prompt_parts.append(str(item.get("text", "")))
                elif item_type == "image_url":
                    image_url = item.get("image_url") or {}
                    if isinstance(image_url, dict):
                        url = image_url.get("url")
                    else:
                        url = image_url
                    if url:
                        image_data.append(url)
                        prompt_parts.append("<image>")
                elif item_type == "image":
                    url = item.get("image")
                    if url:
                        image_data.append(url)
                        prompt_parts.append("<image>")
                else:
                    prompt_parts.append(str(item))
        else:
            prompt_parts.append(str(content))

    prompt_parts.append("<assistant>")
    return "\n".join(part for part in prompt_parts if part), image_data


def _normalize_extract_url(raw_url: str, backend: str) -> str:
    url = raw_url.strip().rstrip("/")
    if backend == "sglang_generate":
        return url if url.endswith("/generate") else f"{url}/generate"
    # openai / vllm backend: always use /v1/chat/completions
    if re.search(r"/(v1/)?chat/completions$", url):
        return url
    # Strip a stray /generate suffix so vLLM never receives a /generate request.
    url = re.sub(r"/generate$", "", url)
    # Avoid double /v1 if base URL already ends with /v1.
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


async def call_extract_model_async(
    *,
    url: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    proxies: dict | None = None,
    executor: ThreadPoolExecutor | None = None,
    source: str = "Extract",
) -> str | None:
    """Call a local extract model served by SGLang or an OpenAI-compatible endpoint.

    Supported modes:
      - EXTRACT_BACKEND=openai or default (incl. vllm): POST /v1/chat/completions
      - EXTRACT_BACKEND=sglang_generate: POST /generate
    """
    try:
        import requests
    except ImportError:
        log_tool_event(
            source,
            "DependencyMissing",
            "'requests' package not installed",
            level="WARNING",
        )
        return None

    backend = os.getenv("EXTRACT_BACKEND", "openai").strip().lower()
    if backend in {"sglang", "generate", "native"}:
        backend = "sglang_generate"

    extract_url = _normalize_extract_url(url, backend)
    timeout = int(os.getenv("EXTRACT_TIMEOUT", "120"))
    temperature = float(os.getenv("EXTRACT_TEMPERATURE", "0"))

    if backend == "sglang_generate":
        prompt, image_data = _extract_message_text_and_images(messages)
        payload: dict[str, Any] = {
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
            },
        }
        if image_data:
            payload["image_data"] = image_data
    else:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

    headers = {"Content-Type": "application/json"}

    try:
        response = await run_with_retries_async(
            lambda: requests.post(
                extract_url,
                headers=headers,
                json=payload,
                timeout=timeout,
                proxies=proxies,
            ),
            executor=executor,
        )
    except Exception as exc:
        log_tool_event(
            source,
            "RequestError",
            f"url={extract_url}",
            error=str(exc),
            level="ERROR",
        )
        return None

    if response.status_code != 200:
        log_tool_event(
            source,
            "HTTPError",
            f"url={extract_url} status={response.status_code}",
            error=response.text,
            level="WARNING",
        )
        return None

    try:
        result = response.json()
    except json.JSONDecodeError:
        log_tool_event(
            source,
            "ParseError",
            "Non-JSON response",
            error=response.text,
            level="WARNING",
        )
        return None

    if not isinstance(result, dict):
        return None

    choices = result.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0] or {}
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(first_choice.get("text"), str):
                return first_choice["text"]

    if isinstance(result.get("text"), str):
        return result["text"]
    if isinstance(result.get("content"), str):
        return result["content"]

    return None


class DeepResearchTool(ABC):
    """Base class for all DeepResearch tools."""

    def __init__(self, name: str, description: str, parameters: dict | None = None):
        self.name = name
        self.description = description
        self.parameters = parameters or {"type": "object", "properties": {}, "required": []}
        self.executor: ThreadPoolExecutor | None = None
        
        # Tool schema for OpenAI function calling
        self._json = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
                or {"type": "object", "properties": {}, "required": []},
            },
        }

    def set_executor(self, executor: ThreadPoolExecutor | None) -> None:
        """Bind a tool executor for blocking calls."""
        self.executor = executor

    def _get_requests_proxies(self) -> dict | None:
        """Build requests-compatible proxy mapping from environment."""
        proxy_value = os.getenv("TOOL_HTTPS_PROXY")
        if proxy_value is None:
            return None
        proxy_value = proxy_value.strip()
        if not proxy_value or proxy_value.lower() == "none":
            return {"http": None, "https": None}
        return {"http": proxy_value, "https": proxy_value}

    async def _run_blocking(self, func: Callable[[], T]) -> T:
        """Run a blocking function in the bound executor."""
        return await run_blocking(func, executor=self.executor)

    @abstractmethod
    async def call(self, **kwargs) -> str:
        """Execute the tool with given arguments."""
        pass

