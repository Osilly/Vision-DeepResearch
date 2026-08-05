"""
Vision-DeepResearch Gym-style interaction environment.

Mirrors the eval rollout (examples/vision_deepresearch/eval/run_eval.py) so RL
and eval share one source of truth for:
  - system prompt build (tools + date substituted at call time),
  - first user observation build (image_id labels + frames + question),
  - multi-turn tool execution loop,
  - terminal reward from an LLM judge.

The interface follows the ms-swift Gym shape (reset / step / close) with
slime-native objects:

    async reset(sample)            -> (first_user_obs, info, system_message)
    async step(response_text)      -> (next_obs, reward, done, info)
    async compute_final_reward(t)  -> float        # for non-natural exits
    async close()                  -> None

`step` returns reward > 0 only on the turn that emits <answer>...</answer>
(judge verdict). Tool turns return reward=0. When the outer rollout exits via
budget/length/max_turns without a natural done=True, call
`compute_final_reward(last_response_text)` to score whatever the model managed
to write.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import json5
import logging
import os
import re
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from examples.vision_deepresearch.base_env import BaseInteractionEnv
from examples.vision_deepresearch.tools.registry import get_tools

logger = logging.getLogger(__name__)

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

_SYSTEM_PROMPT_PATH = Path(__file__).parent / "eval" / "eval_system_prompt.txt"
_SYSTEM_PROMPT_TEMPLATE: str | None = None
_TOOL_MAP: dict[str, dict] | None = None


def _system_prompt_template() -> str:
    global _SYSTEM_PROMPT_TEMPLATE
    if _SYSTEM_PROMPT_TEMPLATE is None:
        _SYSTEM_PROMPT_TEMPLATE = _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    return _SYSTEM_PROMPT_TEMPLATE


def _tool_map() -> dict[str, dict]:
    global _TOOL_MAP
    if _TOOL_MAP is None:
        _TOOL_MAP = {t["function"]["name"]: t for t in get_tools()}
    return _TOOL_MAP


def build_system_message(supported_tool_names: list[str]) -> str:
    tmap = _tool_map()
    tools_json = "\n".join(
        json.dumps(tmap[name], ensure_ascii=False)
        for name in supported_tool_names
        if name in tmap
    )
    date_str = datetime.date.today().isoformat()
    return _system_prompt_template().replace("{tools}", tools_json).replace("{date}", date_str)


def build_user_observation(image_paths: list[str], question: str) -> dict[str, Any]:
    """Tool-calling mode user message: image_id labels + frames + question.

    Mirrors eval.run_eval._build_messages (user half).
    """
    if image_paths:
        content: list[dict] = [
            {"type": "text", "text": "The following are frames sampled from a video clip.\n"}
        ]
        for i, path in enumerate(image_paths):
            content.append({"type": "text", "text": f"image_id: image_{i}\n"})
            content.append({"type": "image", "image": path})
        content.append({"type": "text", "text": f"\n\nQuestion: {question}"})
    else:
        content = [{"type": "text", "text": question}]
    return {"role": "user", "content": content}


class DeepResearchEnv(BaseInteractionEnv):
    """Gym-style env wrapping the eval rollout (tool loop + judge reward).

    Constructed via ``build_env(sample, args)`` (the slime hook). The env keeps
    its own message + image history, so ``step`` only needs the latest assistant
    text. Reward is owned by the env (judge call on <answer> or on terminal text
    via ``compute_final_reward``).
    """

    def __init__(
        self,
        *,
        args: Any,
        max_turns: int,
        work_dir: str = "/tmp/vision_deepresearch",
        supported_tool_names: list[str] | None = None,
        max_consecutive_errors: int = 3,
        sample: Any = None,
    ):
        self.args = args
        self.max_turns = max_turns
        self.work_dir = work_dir
        self.max_consecutive_errors = max_consecutive_errors
        os.makedirs(work_dir, exist_ok=True)

        self.supported_tool_names = supported_tool_names or [
            "search", "visit", "select_crop_search"
        ]

        # Per-episode state (populated by reset)
        self.turn = 0
        self.question: str = ""
        self._label: str | None = None
        self._all_image_paths: list[str] = []
        self._messages_history: list[dict[str, Any]] = []
        self._tool_call_log: list[dict[str, Any]] = []
        self._consecutive_errors = 0
        # Sample stashed by build_env(sample=...) for the sync reset() path
        # (eval calls `env.reset()` with no args); reset_async accepts an
        # explicit sample and overrides this.
        self._initial_sample = sample
        # Event loop reused by sync step() for tool execution.
        self._sync_loop: asyncio.AbstractEventLoop | None = None

        # Tool instances (lazy)
        self._search_tool = None
        self._visit_tool = None
        self._crop_and_search_tool = None
        self._select_crop_search_tool = None

    # ------------------------------------------------------------------
    # Gym interface (async, used by RL rollout — reset_async/step_async
    # own the per-turn judge reward). Eval uses the sync reset()/step()
    # further below.
    # ------------------------------------------------------------------

    async def reset_async(self, sample: Any = None) -> tuple[dict[str, Any], dict[str, Any], str]:
        """Async gym-style reset. Returns (first_user_observation, info, system_message).

        Pulls images from sample.metadata['raw_prompt'] (preferred — set by
        slime's data pipeline) or sample.multimodal_inputs['images'] (fallback).
        Pulls question from sample.metadata['question'] / sample.prompt (str)
        / extracted from raw_prompt text. Pulls ground truth from sample.label.
        """
        if sample is None:
            sample = self._initial_sample
        return self._reset_state(sample)

    async def step_async(
        self, response_text: str
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Process one assistant turn.

        Returns (observation, reward, done, info).
          - <answer> tag       → judge call → (empty obs, reward, True, info)
          - tool call          → execute    → (tool_response obs, 0.0, False, info)
          - parse fail / unknown tool → (error obs, 0.0, terminal?, info)
        """
        self.turn += 1
        is_final_turn = self.turn >= self.max_turns
        info: dict[str, Any] = {}

        self._messages_history.append({"role": "assistant", "content": response_text})

        # 1) <answer> → end with judge reward.
        answer = _extract_answer(response_text)
        if answer:
            info["final_answer"] = answer
            info["tool_executed"] = False
            reward = await self._judge(response_text)
            info["judge_reward"] = reward
            obs = {"obs_str": "", "role": "user", "multi_modal_data": {}}
            return obs, float(reward), True, info

        # 2) tool call.
        tool_call = _extract_tool_call(response_text)
        info["tool_call"] = deepcopy(tool_call)

        if not tool_call:
            self._consecutive_errors += 1
            info["tool_executed"] = False
            obs = {
                "obs_str": _format_tool_response(
                    "No <tool_call> or <answer> detected. "
                    f"Available tools: {', '.join(self.supported_tool_names)}. "
                    "If you have gathered enough information, provide your answer in "
                    "<answer></answer> tags. Otherwise call a tool via "
                    "<tool_call>{\"name\": \"<name>\", \"arguments\": {...}}</tool_call>."
                ),
                "role": "user",
                "multi_modal_data": {},
            }
            done = is_final_turn or self._consecutive_errors >= self.max_consecutive_errors
            return obs, 0.0, done, info

        name = tool_call["name"]
        if name not in self.supported_tool_names:
            self._consecutive_errors += 1
            info["tool_executed"] = False
            obs = {
                "obs_str": _format_tool_response(
                    f"Tool `{name}` is not supported. "
                    f"Available tools: {', '.join(self.supported_tool_names)}."
                ),
                "role": "user",
                "multi_modal_data": {},
            }
            done = is_final_turn or self._consecutive_errors >= self.max_consecutive_errors
            return obs, 0.0, done, info

        # 3) Execute tool.
        try:
            result = await self._execute_tool_async(tool_call)
        except Exception as exc:
            logger.warning("Tool execution failed: %s", exc)
            result = {"error": f"Tool execution failed: {exc}"}

        info["tool_result"] = result
        info["tool_executed"] = True
        self._consecutive_errors = 0

        if "error" in result:
            obs_content = f"Tool `{name}` returned:\n{result['error']}"
        elif "result" in result:
            obs_content = f"Tool `{name}` returned:\n{result['result']}"
        else:
            obs_content = f"Tool `{name}` returned:\n{result}"

        obs_str = _format_tool_response(obs_content)
        self._messages_history.append({"role": "user", "content": obs_str})
        self._tool_call_log.append({
            "turn": self.turn,
            "name": name,
            "arguments": tool_call.get("arguments"),
        })

        obs = {"obs_str": obs_str, "role": "tool", "multi_modal_data": {}}
        return obs, 0.0, is_final_turn, info

    async def compute_final_reward(self, response_text: str) -> float:
        """Judge `response_text` regardless of whether it contains <answer>.

        Called by the outer rollout when the loop exits via budget / max_turns
        / finish_type length|abort — so a model that ran out of room still gets
        credit for the best partial answer it produced.
        """
        if not response_text:
            return 0.0
        return await self._judge(response_text)

    async def close(self) -> None:
        return

    # ------------------------------------------------------------------
    # Sync interface (backwards-compat for eval/run_eval.py which calls
    # env.reset() with no args and env.step(text) via asyncio.to_thread).
    # These paths do NOT invoke the LLM judge — reward is computed
    # externally by run_eval.py's reward hook.
    # ------------------------------------------------------------------

    def reset(self, sample: Any = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Sync reset. Returns (first_user_observation, info) — 2-tuple to
        match the legacy eval interface. Uses `sample` from build_env if not
        passed explicitly (eval's `_generate_*` do it that way)."""
        if sample is None:
            sample = self._initial_sample
        obs, info, _system_message = self._reset_state(sample)
        return obs, info

    def step(
        self, response_text: str
    ) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        """Sync step. Returns (observation, done, info) — legacy 3-tuple, no
        reward. `<answer>` marks the episode as done but no judge call is
        made here (reward is computed externally by run_eval.py)."""
        self.turn += 1
        is_final_turn = self.turn >= self.max_turns
        info: dict[str, Any] = {}

        self._messages_history.append({"role": "assistant", "content": response_text})

        # 1) <answer> → done, no reward here.
        answer = _extract_answer(response_text)
        if answer:
            info["final_answer"] = answer
            info["tool_executed"] = False
            obs = {"obs_str": "", "role": "user", "multi_modal_data": {}}
            return obs, True, info

        # 2) parse tool call.
        tool_call = _extract_tool_call(response_text)
        info["tool_call"] = deepcopy(tool_call)

        if not tool_call:
            self._consecutive_errors += 1
            info["tool_executed"] = False
            obs = {
                "obs_str": _format_tool_response(
                    "No <tool_call> or <answer> detected. "
                    f"Available tools: {', '.join(self.supported_tool_names)}. "
                    "If you have gathered enough information, provide your answer in "
                    "<answer></answer> tags. Otherwise call a tool via "
                    "<tool_call>{\"name\": \"<name>\", \"arguments\": {...}}</tool_call>."
                ),
                "role": "user",
                "multi_modal_data": {},
            }
            done = is_final_turn or self._consecutive_errors >= self.max_consecutive_errors
            return obs, done, info

        name = tool_call["name"]
        if name not in self.supported_tool_names:
            self._consecutive_errors += 1
            info["tool_executed"] = False
            obs = {
                "obs_str": _format_tool_response(
                    f"Tool `{name}` is not supported. "
                    f"Available tools: {', '.join(self.supported_tool_names)}."
                ),
                "role": "user",
                "multi_modal_data": {},
            }
            done = is_final_turn or self._consecutive_errors >= self.max_consecutive_errors
            return obs, done, info

        # 3) Execute tool via a private event loop (safe when this method
        # is invoked via asyncio.to_thread from an outer async context).
        try:
            if self._sync_loop is None or self._sync_loop.is_closed():
                self._sync_loop = asyncio.new_event_loop()
            result = self._sync_loop.run_until_complete(self._execute_tool_async(tool_call))
        except Exception as exc:
            logger.warning("Tool execution failed: %s", exc)
            result = {"error": f"Tool execution failed: {exc}"}

        info["tool_result"] = result
        info["tool_executed"] = True
        self._consecutive_errors = 0

        if "error" in result:
            obs_content = f"Tool `{name}` returned:\n{result['error']}"
        elif "result" in result:
            obs_content = f"Tool `{name}` returned:\n{result['result']}"
        else:
            obs_content = f"Tool `{name}` returned:\n{result}"

        obs_str = _format_tool_response(obs_content)
        self._messages_history.append({"role": "user", "content": obs_str})
        self._tool_call_log.append({
            "turn": self.turn,
            "name": name,
            "arguments": tool_call.get("arguments"),
        })

        obs = {"obs_str": obs_str, "role": "tool", "multi_modal_data": {}}
        return obs, is_final_turn, info

    # ------------------------------------------------------------------
    # Shared setup used by both reset() (sync) and reset_async() (async).
    # ------------------------------------------------------------------

    def _reset_state(self, sample: Any) -> tuple[dict[str, Any], dict[str, Any], str]:
        self.turn = 0
        self._messages_history.clear()
        self._tool_call_log.clear()
        self._consecutive_errors = 0

        images = _extract_image_paths_from_sample(sample) if sample is not None else []
        self._all_image_paths = list(images)

        self.question = _extract_question_from_sample(sample) if sample is not None else ""
        self._label = _extract_ground_truth(sample) if sample is not None else None

        system_message = build_system_message(self.supported_tool_names)
        user_obs = build_user_observation(images, self.question)

        self._messages_history.append({"role": "system", "content": system_message})
        self._messages_history.append(user_obs)

        info = {
            "supported_tools": list(self.supported_tool_names),
            "num_images": len(images),
            "has_label": self._label is not None,
        }
        return user_obs, info, system_message

    # ------------------------------------------------------------------
    # Backwards-compatible sync wrappers (only for code paths that still
    # use asyncio.to_thread(env.step, ...)). New code should `await`.
    # ------------------------------------------------------------------

    def format_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Convert env-step observation dict → chat message dict."""
        obs_str = observation.get("obs_str", "") or ""
        return {"role": "user", "content": [{"type": "text", "text": obs_str}]}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _judge(self, response_text: str) -> float:
        """Call the LLM judge via the shared rm_hub helper."""
        from slime.rollout.rm_hub.multimodal import call_llm_judge

        if self._label is None:
            logger.warning("Judge called but sample.label is None; returning 0.0")
            return 0.0

        sample_shim = SimpleNamespace(
            response=response_text,
            label=self._label,
            metadata={"prompt": self.question},
        )
        try:
            return float(await call_llm_judge(self.args, sample_shim))
        except Exception as exc:
            logger.warning("Judge call failed: %s", exc)
            return 0.0

    async def _execute_tool_async(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        name = tool_call["name"]
        arguments = tool_call.get("arguments") or {}

        if name == "search":
            if "query" not in arguments:
                return {"error": "search tool requires 'query' argument"}
            return {"result": await self._search.call(**arguments)}

        if name == "visit":
            if "url" not in arguments:
                return {"error": "visit tool requires 'url' argument"}
            return {"result": await self._visit.call(**arguments)}

        if name == "crop_and_search":
            if "image_id" not in arguments and self._all_image_paths:
                arguments["image_id"] = self._all_image_paths[0]
            elif "image_id" in arguments:
                raw_image_id = arguments["image_id"]
                resolved_image_id = self._resolve_image_id(raw_image_id)
                if (
                    isinstance(raw_image_id, str)
                    and re.fullmatch(r"image_(\d+)", raw_image_id)
                    and resolved_image_id == raw_image_id
                ):
                    available = [f"image_{i}" for i in range(len(self._all_image_paths))]
                    if available:
                        return {"error": (
                            f"Requested image_id `{raw_image_id}` is not available. "
                            f"Available image ids: {', '.join(available)}."
                        )}
                    return {"error": (
                        f"Requested image_id `{raw_image_id}` is not available "
                        "because no images are registered."
                    )}
                arguments["image_id"] = resolved_image_id
            return {"result": await self._crop_search.call(**arguments)}

        if name == "select_crop_search":
            image_idx_raw = arguments.get("image_idx", [])
            if not image_idx_raw:
                return {"error": "select_crop_search requires 'image_idx' argument."}
            resolved_paths = []
            for raw_idx in image_idx_raw:
                try:
                    idx = int(raw_idx)
                except (ValueError, TypeError):
                    return {"error": f"image_idx values must be integers, got {raw_idx!r}."}
                if not (0 <= idx < len(self._all_image_paths)):
                    return {"error": (
                        f"image_idx {idx} is out of range. "
                        f"Available indices: {list(range(len(self._all_image_paths)))}."
                    )}
                resolved_paths.append(self._all_image_paths[idx])

            bbox_list = arguments.get("bbox", [])
            if not bbox_list:
                return {"error": "select_crop_search requires 'bbox' argument."}
            if len(bbox_list) != len(resolved_paths):
                return {"error": (
                    f"bbox and image_idx must have the same length "
                    f"(got {len(bbox_list)} bboxes for {len(resolved_paths)} images)."
                )}
            return {"result": await self._select_crop_search.call(
                image_paths=resolved_paths,
                bbox=bbox_list,
                goal=arguments.get("goal", ""),
            )}

        return {"error": f"Unknown tool: {name}"}

    def _resolve_image_id(self, image_id: Any) -> Any:
        if not isinstance(image_id, str):
            return image_id
        match = re.fullmatch(r"image_(\d+)", image_id)
        if match:
            idx = int(match.group(1))
            if 0 <= idx < len(self._all_image_paths):
                return self._all_image_paths[idx]
        return image_id

    # Lazy tool properties
    @property
    def _search(self):
        if self._search_tool is None:
            from examples.vision_deepresearch.tools.search_tool import SearchTool
            self._search_tool = SearchTool()
        return self._search_tool

    @property
    def _visit(self):
        if self._visit_tool is None:
            from examples.vision_deepresearch.tools.visit_tool import VisitTool
            self._visit_tool = VisitTool()
        return self._visit_tool

    @property
    def _crop_search(self):
        if self._crop_and_search_tool is None:
            from examples.vision_deepresearch.tools.crop_and_search_tool import CropAndSearchTool
            self._crop_and_search_tool = CropAndSearchTool()
        return self._crop_and_search_tool

    @property
    def _select_crop_search(self):
        if self._select_crop_search_tool is None:
            from examples.vision_deepresearch.tools.select_crop_search_tool import SelectCropSearchTool
            self._select_crop_search_tool = SelectCropSearchTool()
        return self._select_crop_search_tool


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _extract_answer(text: str) -> str | None:
    m = ANSWER_RE.search(text)
    return m.group(1).strip() if m else None


def _extract_tool_call(text: str) -> dict[str, Any] | None:
    matches = list(TOOL_CALL_RE.finditer(text))
    if not matches:
        return None
    raw = matches[-1].group(1).strip()
    try:
        payload = json5.loads(raw)
    except ValueError as exc:
        logger.warning("Failed to decode tool call payload: %s", exc)
        return None
    name = payload.get("name") or payload.get("function", {}).get("name")
    arguments = payload.get("arguments") or payload.get("function", {}).get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json5.loads(arguments)
        except ValueError:
            return None
    if not name:
        return None
    return {"name": name, "arguments": arguments}


def _format_tool_response(content: str) -> str:
    return f"<tool_response>\n{content}\n</tool_response>"


def _extract_image_paths_from_sample(sample: Any) -> list[str]:
    """Resolve image paths: prefer raw_prompt, fall back to multimodal_inputs."""
    raw_prompt = None
    if getattr(sample, "metadata", None):
        raw_prompt = sample.metadata.get("raw_prompt")
    if raw_prompt:
        paths = _extract_image_paths_from_raw_prompt(raw_prompt)
        if paths:
            return paths
    mm = getattr(sample, "multimodal_inputs", None) or {}
    return [p for p in (mm.get("images") or []) if isinstance(p, str)]


def _extract_image_paths_from_raw_prompt(raw_prompt: Any) -> list[str]:
    image_paths: list[str] = []

    def visit(value: Any):
        if isinstance(value, dict):
            image = value.get("image")
            if isinstance(image, str):
                image_paths.append(image)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(raw_prompt)
    return image_paths


def _extract_question_from_sample(sample: Any) -> str:
    """Prefer sample.metadata['question'], then sample.prompt (str), then dig
    out the trailing 'Question: ...' text from raw_prompt user content.
    """
    if getattr(sample, "metadata", None):
        q = sample.metadata.get("question")
        if q:
            return str(q)
    prompt = getattr(sample, "prompt", None)
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    raw_prompt = (sample.metadata or {}).get("raw_prompt") if getattr(sample, "metadata", None) else None
    if isinstance(raw_prompt, list):
        for msg in reversed(raw_prompt):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for part in reversed(content):
                    txt = part.get("text") if isinstance(part, dict) else None
                    if isinstance(txt, str) and "Question:" in txt:
                        return txt.split("Question:", 1)[1].strip()
                    if isinstance(txt, str) and txt.strip():
                        return txt.strip()
    return ""


def _extract_ground_truth(sample: Any) -> str | None:
    if sample is None:
        return None
    label = getattr(sample, "label", None)
    if label is not None:
        return str(label)
    return None


def _sync_tool_config_to_env(args=None) -> None:
    """Mirror YAML/args tool config onto environment variables consumed by tools."""
    if args is None:
        return
    mapping = {
        "zhipu_api_key": "ZHIPU_API_KEY",
        "serp_api_key": "SERP_API_KEY",
        "jina_api_key": "JINA_API_KEY",
        "oss_access_key_id": "OSS_ACCESS_KEY_ID",
        "oss_access_key_secret": "OSS_ACCESS_KEY_SECRET",
        "oss_endpoint": "OSS_ENDPOINT",
        "oss_bucket_name": "OSS_BUCKET_NAME",
        "image_crop_cache": "IMAGE_CROP_CACHE",
        "extract_model": "EXTRACT_MODEL",
        "extract_backend": "EXTRACT_BACKEND",
        "extract_url": "EXTRACT_URL",
        "extract_timeout": "EXTRACT_TIMEOUT",
        "extract_temperature": "EXTRACT_TEMPERATURE",
        "text_search_url": "TEXT_SEARCH_URL",
        "image_search_url": "IMAGE_SEARCH_URL",
        "reader_url": "READER_URL",
        "tool_https_proxy": "TOOL_HTTPS_PROXY",
    }
    for attr, env_name in mapping.items():
        value = getattr(args, attr, None)
        if value is None:
            continue
        os.environ[env_name] = str(value)


# ---------------------------------------------------------------------------
# Slime hook: factory called by rollout.py
# ---------------------------------------------------------------------------

def build_env(sample=None, args=None, **_: Any):
    """Construct a DeepResearchEnv from sample metadata and CLI args.

    Expected args attributes (typically loaded from --custom-config-path YAML):
        max_turns:           int
        vlm_rollout_work_dir: str (default /tmp/vision_deepresearch)
        supported_tools:     str (comma-separated) | list[str]
        judge_url:           str (for terminal reward via call_llm_judge)
        extract_url / extract_model / zhipu_api_key / ... (mirrored to env)
    """
    _sync_tool_config_to_env(args)

    max_turns = getattr(args, "max_turns", None)
    if max_turns is None:
        raise ValueError(
            "max_turns must be set via --custom-config-path in the custom config file."
        )
    work_dir = getattr(args, "vlm_rollout_work_dir", "/tmp/vision_deepresearch")

    supported_tools = getattr(args, "supported_tools", None)
    if isinstance(supported_tools, str):
        supported_tools = [t.strip() for t in supported_tools.split(",") if t.strip()]

    return DeepResearchEnv(
        args=args,
        max_turns=max_turns,
        work_dir=work_dir,
        supported_tool_names=supported_tools,
        sample=sample,
    )
