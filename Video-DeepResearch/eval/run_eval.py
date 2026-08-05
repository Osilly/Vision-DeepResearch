#!/usr/bin/env python3
"""
VideoDR evaluation script.

Reads the benchmark CSV, finds extracted frames for each video, runs the
same multi-turn rollout pipeline as the data producer, then reports accuracy
broken down by category and difficulty.

Input CSV columns (no header):
  video_id, question, answer, category, difficulty

Frames must already be extracted by extract_keyframes.py:
  {frames_dir}/{video_id}/frame_XXXX.XX.png

Usage:
  python3 run_eval.py \
    --csv       /path/to/VideoDR.csv \
    --frames-dir /path/to/frames \
    --config    /path/to/config.yaml \
    --system-prompt-file /path/to/eval_system_prompt.txt \
    --hf-checkpoint /path/to/model \
    --sglang-url http://localhost:13141 \
    --reward-model-url http://localhost:13141 \
    --output-dir /path/to/eval/output \
    --max-async-samples 4 \
    --max-turns 20 \
    --max-new-tokens 8192
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import datetime
import io
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vdr_core.deepresearch_reward import DeepResearchReward
from vdr_core.rollout import generate
from vdr_core.tools.registry import get_tools
from vdr_core.slime_utils.processing_utils import load_processor, load_tokenizer, process_vision_info
from vdr_core.slime_utils.types import Sample

log = logging.getLogger(__name__)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
END_TOKEN_RE = re.compile(r"(?:<\|?im_end\|?>|<\|endoftext\|>|<\|eot_id\|>)\s*$")
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

_TOOL_MAP = {t["function"]["name"]: t for t in get_tools()}


# ---------------------------------------------------------------------------
# Config wrapper (same as data producer)
# ---------------------------------------------------------------------------

class Args:
    def __init__(self, c: dict[str, Any]):
        self.hf_checkpoint = c["hf_checkpoint"]
        self.apply_chat_template = c.get("apply_chat_template", True)
        self.apply_chat_template_kwargs = c.get("apply_chat_template_kwargs", {}) or {}
        self.input_key = c.get("input_key", "prompt")
        self.label_key = c.get("label_key", "label")
        self.metadata_key = c.get("metadata_key", "metadata")
        self.tool_key = c.get("tool_key")
        self.multimodal_keys = c.get("multimodal_keys", {"image": "images"})
        self.rollout_max_prompt_len = c.get("rollout_max_prompt_len")
        self.multimodal_load_workers = c.get("multimodal_load_workers", 0)
        self.sglang_router_ip = c.get("sglang_router_ip", "localhost")
        self.sglang_router_port = int(c.get("sglang_router_port", 13141))
        self.sglang_server_concurrency = int(c.get("sglang_server_concurrency", 16))
        self.rollout_num_gpus = int(c.get("rollout_num_gpus", 1))
        self.rollout_num_gpus_per_engine = int(c.get("rollout_num_gpus_per_engine", 1))
        self.partial_rollout = False
        self.max_turns = c.get("max_turns", 20)
        self.rollout_max_context_len = c.get("rollout_max_context_len")
        self.vlm_rollout_work_dir = c.get("vlm_rollout_work_dir", "/tmp/video_eval")
        self.rollout_interaction_env_path = c.get(
            "rollout_interaction_env_path", "vdr_core.env"
        )
        self.rollout_temperature = c.get("rollout_temperature", 0.0)
        self.rollout_top_p = c.get("rollout_top_p", 0.9)
        self.rollout_top_k = c.get("rollout_top_k", 20)
        self.rollout_max_response_len = c.get("rollout_max_response_len", 32000)
        self.rollout_stop = c.get("rollout_stop")
        self.rollout_stop_token_ids = c.get("rollout_stop_token_ids")
        self.rollout_skip_special_tokens = c.get("rollout_skip_special_tokens", False)
        self.sglang_enable_deterministic_inference = c.get(
            "sglang_enable_deterministic_inference", False
        )
        self.rollout_seed = c.get("rollout_seed", 42)
        self.n_samples_per_prompt = c.get("n_samples_per_prompt", 1)
        for k, v in c.items():
            if not hasattr(self, k):
                setattr(self, k, v)


# ---------------------------------------------------------------------------
# System prompt building
# ---------------------------------------------------------------------------

def _build_system_content(template: str, supported_tool_names: list[str]) -> str:
    tools_json = "\n".join(
        json.dumps(_TOOL_MAP[name], ensure_ascii=False)
        for name in supported_tool_names
        if name in _TOOL_MAP
    )
    date_str = datetime.date.today().isoformat()
    return template.replace("{tools}", tools_json).replace("{date}", date_str)


def _build_direct_system_content(template: str) -> str:
    date_str = datetime.date.today().isoformat()
    return template.replace("{date}", date_str)


# ---------------------------------------------------------------------------
# CSV / JSONL loading
# ---------------------------------------------------------------------------

def load_csv(csv_path: str) -> list[dict]:
    """Load VideoDR.csv; columns: video_id, question, answer, category, difficulty."""
    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or not row[0].strip():
                continue
            records.append({
                "video_id":   row[0].strip(),
                "question":   row[1].strip(),
                "answer":     row[2].strip(),
                "category":   row[3].strip() if len(row) > 3 else "",
                "difficulty": row[4].strip() if len(row) > 4 else "",
                "images":     None,  # CSV always resolves frames via find_frames()
            })
    return records


def load_jsonl(jsonl_path: str) -> list[dict]:
    """Load a JSONL benchmark file.

    Expected fields per row (extras are ignored):
      id | video_id         -> video_id  (str)
      question              -> question  (str)
      label  | answer       -> answer    (str)  ground truth
      category              -> category  (optional)
      difficulty            -> difficulty (optional)
      images                -> list of frame paths (optional; falls back to
                               find_frames(frames_dir, video_id) when missing/empty)
    """
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{jsonl_path}: bad JSON on line {ln}: {exc}")
            vid = row.get("video_id") or row.get("id")
            q   = row.get("question")
            ans = row.get("label") if row.get("label") is not None else row.get("answer")
            if not vid or not q or ans is None:
                log.warning("JSONL line %d missing video_id/question/label — skipping", ln)
                continue
            imgs = row.get("images")
            if imgs is not None and not isinstance(imgs, list):
                imgs = [imgs]
            records.append({
                "video_id":   str(vid).strip(),
                "question":   str(q).strip(),
                "answer":     str(ans).strip(),
                "category":   str(row.get("category") or "").strip(),
                "difficulty": str(row.get("difficulty") or "").strip(),
                "images":     imgs,
            })
    return records


def find_frames(frames_dir: str, video_id: str) -> list[str]:
    """Return sorted frame paths for a video (by filename = timestamp order)."""
    vid_dir = Path(frames_dir) / video_id
    if not vid_dir.exists():
        return []
    return sorted(str(p) for p in vid_dir.glob("frame_*.png"))


# ---------------------------------------------------------------------------
# Sample building (same logic as data producer)
# ---------------------------------------------------------------------------

def _build_messages(image_paths: list[str], question: str, system_content: str) -> list[dict]:
    """Tool-calling mode: annotate each frame with image_id for tool references."""
    if image_paths:
        user_content: list = [
            {"type": "text", "text": "The following are frames sampled from a video clip.\n"}
        ]
        for i, path in enumerate(image_paths):
            user_content.append({"type": "text", "text": f"image_id: image_{i}\n"})
            user_content.append({"type": "image", "image": path})
        user_content.append({"type": "text", "text": f"\n\nQuestion: {question}"})
    else:
        user_content = question

    return [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": user_content},
    ]


def _build_messages_direct(image_paths: list[str], question: str, system_content: str) -> list[dict]:
    """Direct-answer mode: simple frame display + question, no tool references."""
    if image_paths:
        user_content: list = [
            {"type": "text", "text": "The following are keyframes sampled from a video clip:\n"}
        ]
        for path in image_paths:
            user_content.append({"type": "image", "image": path})
        user_content.append({"type": "text", "text": f"\n\nQuestion: {question}"})
    else:
        user_content = question

    return [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": user_content},
    ]


def _build_sample(
    image_paths: list[str],
    question: str,
    label: str,
    system_content: str,
    metadata: dict,
    tokenizer,
    processor,
    build_messages_fn=None,
) -> Sample:
    if build_messages_fn is None:
        build_messages_fn = _build_messages
    messages = build_messages_fn(image_paths, question, system_content)
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    mm_inputs = None
    if processor and image_paths:
        mm_inputs = process_vision_info(messages, processor)

    return Sample(
        prompt=formatted,
        label=label,
        metadata={**metadata, "raw_prompt": messages},
        multimodal_inputs=mm_inputs,
    )


# ---------------------------------------------------------------------------
# Utilities (same as data producer)
# ---------------------------------------------------------------------------

def _json_safe(x):
    if isinstance(x, Sample.Status): return x.value
    if isinstance(x, dict): return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [_json_safe(v) for v in x]
    if hasattr(x, "tolist"): return _json_safe(x.tolist())
    try:
        json.dumps(x); return x
    except TypeError:
        return str(x)


def parse_url(u: str):
    p = urlparse(u if "://" in u else "http://" + u)
    return p.hostname, p.port or 80


def load_existing_indices(path: str) -> set[int]:
    output = Path(path)
    if not output.exists():
        return set()
    indices: set[int] = set()
    with output.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                log.warning("skip malformed line %s in %s", line_no, output)
                continue
            idx = rec.get("index")
            if isinstance(idx, int):
                indices.add(idx)
    return indices


def clean_end_tokens(text):
    if not isinstance(text, str):
        return text
    prev = None
    while prev != text:
        prev = text
        text = END_TOKEN_RE.sub("", text).rstrip()
    return text


def clean_trajectory(x):
    if isinstance(x, list):
        return [clean_trajectory(v) for v in x]
    if isinstance(x, dict):
        y = {k: clean_trajectory(v) for k, v in x.items()}
        if isinstance(y.get("content"), str):
            y["content"] = clean_end_tokens(y["content"])
        return y
    return clean_end_tokens(x)


def ans(text):
    ms = list(ANSWER_RE.finditer(text or ""))
    return ms[-1].group(1).strip() if ms else None


def ans_traj(t):
    for m in reversed(t):
        if m.get("role") == "assistant" and isinstance(m.get("content"), str):
            a = ans(m["content"])
            if a: return a
    return None


def _content_to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return "" if content is None else str(content)


def extract_first_user_question(t, fallback):
    for m in t:
        if isinstance(m, dict) and m.get("role") == "user":
            text = _content_to_text(m.get("content"))
            match = re.search(r"Question\s*[:：]\s*([\s\S]*)", text, flags=re.I)
            if match:
                return match.group(1).strip()
            if text.strip():
                return text.strip()
    return json.dumps(fallback, ensure_ascii=False)


def extract_last_assistant_response(t, fallback):
    for m in reversed(t):
        if isinstance(m, dict) and m.get("role") == "assistant":
            text = _content_to_text(m.get("content"))
            if text.strip():
                return clean_end_tokens(text)
    return clean_end_tokens(fallback)


def add_think_prefix(t):
    out = []
    for m in t:
        if not isinstance(m, dict):
            out.append(m); continue
        item = dict(m)
        content = item.get("content")
        if item.get("role") == "assistant" and isinstance(content, str):
            stripped = content.lstrip()
            if stripped and not stripped.lower().startswith("<think>"):
                prefix_space = content[: len(content) - len(stripped)]
                item["content"] = f"{prefix_space}<think>{stripped}"
        out.append(item)
    return out


def turns(t):
    return sum(1 for m in t if m.get("role") == "assistant")


def clone(s: Sample, idx: int):
    x = Sample.from_dict(s.to_dict())
    x.index, x.status = idx, Sample.Status.PENDING
    x.tokens, x.response, x.response_length = [], "", 0
    x.loss_mask, x.rollout_log_probs, x.multimodal_train_inputs = [], [], None
    return x


def sampling_params(a: Args):
    return dict(
        temperature=a.rollout_temperature,
        top_p=a.rollout_top_p,
        top_k=a.rollout_top_k,
        max_new_tokens=a.rollout_max_response_len,
        stop=a.rollout_stop,
        stop_token_ids=a.rollout_stop_token_ids,
        skip_special_tokens=a.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )


# ---------------------------------------------------------------------------
# OpenAI-compatible rollout helpers
# ---------------------------------------------------------------------------

def _to_openai_messages(messages: list[dict]) -> list[dict]:
    """Convert qwen-style messages (with file-path images) to OpenAI chat format."""
    import base64
    import os
    result = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            result.append({"role": msg["role"], "content": content})
        elif isinstance(content, list):
            parts = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                t = item.get("type")
                if t == "text":
                    parts.append({"type": "text", "text": item.get("text", "")})
                elif t == "image":
                    path = item.get("image", "")
                    if path and os.path.exists(path):
                        with open(path, "rb") as f:
                            b64 = base64.b64encode(f.read()).decode()
                        parts.append({"type": "image_url",
                                      "image_url": {"url": f"data:image/png;base64,{b64}"}})
            result.append({"role": msg["role"], "content": parts})
        else:
            result.append({"role": msg["role"], "content": content or ""})
    return result


async def _generate_openai(a: Args, sample: Sample, openai_cfg: dict) -> Sample:
    """Multi-turn rollout using OpenAI-compatible chat completions API."""
    import importlib
    try:
        import openai as _openai
    except ImportError:
        raise RuntimeError("openai package not installed; run: pip install openai")

    client = _openai.AsyncOpenAI(
        api_key=openai_cfg["api_key"],
        base_url=openai_cfg["base_url"],
        default_headers=openai_cfg.get("headers", {}),
    )

    # Build env for tool execution (same as rollout.py)
    env_path = getattr(a, "rollout_interaction_env_path", "vdr_core.env")
    env_module = importlib.import_module(env_path)
    env = env_module.build_env(sample=sample, args=a)
    env.reset()

    raw_prompt = sample.metadata.get("raw_prompt", [])
    messages: list[dict] = [dict(m) for m in raw_prompt]  # working copy (qwen-style)
    trajectory: list[dict] = [dict(m) for m in raw_prompt]

    max_turns = getattr(a, "max_turns", 20) or 20
    last_text = ""

    for _turn in range(max_turns):
        try:
            resp = await client.chat.completions.create(
                model=openai_cfg["model"],
                messages=_to_openai_messages(messages),
                max_tokens=openai_cfg.get("max_tokens", 8192),
                temperature=openai_cfg.get("temperature", 0.0),
                stream=False,
            )
            assistant_text = resp.choices[0].message.content or ""
        except Exception as exc:
            log.error("OpenAI call failed at turn %d: %s", _turn, exc)
            break

        last_text = assistant_text
        msg_asst = {"role": "assistant", "content": assistant_text}
        messages.append(msg_asst)
        trajectory.append(msg_asst)

        if ANSWER_RE.search(assistant_text):
            break

        # Execute tool if present
        obs, done, _info = await asyncio.to_thread(env.step, assistant_text)
        obs_str = obs.get("obs_str", "")
        msg_obs = {"role": "user", "content": obs_str}
        messages.append(msg_obs)
        trajectory.append(msg_obs)

        if done:
            break

    sample.response = last_text
    sample.metadata["trajectory"] = trajectory
    return sample


# ---------------------------------------------------------------------------
# Claude (Bedrock-compatible) rollout helpers
# ---------------------------------------------------------------------------

_DEFAULT_CLAUDE_MAX_IMAGE_BYTES = int(3.6 * 1024 * 1024)
_DEFAULT_CLAUDE_MAX_IMAGE_DIM = 8000


def _claude_encode_image(
    path: str,
    max_dim: int = _DEFAULT_CLAUDE_MAX_IMAGE_DIM,
    max_bytes: int = _DEFAULT_CLAUDE_MAX_IMAGE_BYTES,
) -> tuple[str, str]:
    """Encode an image as base64 JPEG, enforcing dim and size limits."""
    from PIL import Image
    img = Image.open(path)
    if img.mode in ("RGBA", "P", "LA", "CMYK"):
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

    quality = 90
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    data = buf.getvalue()
    while len(data) > max_bytes and quality > 25:
        quality -= 10
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        data = buf.getvalue()

    # Still too big? downscale iteratively.
    while len(data) > max_bytes and max(img.size) > 256:
        img = img.resize(
            (max(1, int(img.width * 0.85)), max(1, int(img.height * 0.85))),
            Image.LANCZOS,
        )
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        data = buf.getvalue()

    return "image/jpeg", base64.b64encode(data).decode()


def _split_claude_system(messages: list[dict]) -> tuple[str | None, dict | None]:
    """Return (system_text, initial_user_msg). Drops anything after the first user."""
    system_parts: list[str] = []
    initial_user: dict | None = None
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            text = content if isinstance(content, str) else _content_to_text(content)
            if text:
                system_parts.append(text)
        elif role == "user" and initial_user is None:
            initial_user = msg
    return ("\n\n".join(system_parts) if system_parts else None), initial_user


async def _build_claude_initial_user(
    user_msg: dict | None, max_dim: int, max_bytes: int
) -> dict:
    """Build the first user message for Claude. Image encoding runs in parallel."""
    if user_msg is None:
        return {"role": "user", "content": [{"type": "text", "text": ""}]}
    content = user_msg.get("content")
    if isinstance(content, str):
        return {"role": "user", "content": [{"type": "text", "text": content}]}

    parts: list[Any] = []
    encode_tasks: list = []
    encode_slots: list[int] = []
    for item in content or []:
        if not isinstance(item, dict):
            continue
        t = item.get("type")
        if t == "text":
            parts.append({"type": "text", "text": item.get("text", "")})
        elif t == "image":
            path = item.get("image", "")
            if path and os.path.exists(path):
                encode_slots.append(len(parts))
                parts.append(None)  # placeholder
                encode_tasks.append(
                    asyncio.to_thread(_claude_encode_image, path, max_dim, max_bytes)
                )

    if encode_tasks:
        t0 = time.time()
        encoded = await asyncio.gather(*encode_tasks)
        log.info("Claude image encode: %d images in %.2fs", len(encode_tasks), time.time() - t0)
        for pos, (media_type, b64) in zip(encode_slots, encoded):
            parts[pos] = {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": b64},
            }

    return {"role": "user", "content": parts or [{"type": "text", "text": ""}]}


def _extract_claude_response(data: Any) -> tuple[str, str, dict | None]:
    """Pull (text, thinking, normalized_payload) out of a Claude response.

    Handles raw Anthropic Messages, Bedrock proxies that wrap the body inside
    `body`/`data`/`result`/`response`/`output`, and string-encoded bodies.
    """
    payload = data
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            return "", "", None
    if not isinstance(payload, dict):
        return "", "", None

    # Unwrap one common envelope layer if `content` isn't at the top.
    if "content" not in payload:
        for key in ("body", "data", "result", "response", "output", "message"):
            sub = payload.get(key)
            if isinstance(sub, str):
                try:
                    sub = json.loads(sub)
                except Exception:
                    continue
            if isinstance(sub, dict) and "content" in sub:
                payload = sub
                break

    blocks = payload.get("content")
    if not isinstance(blocks, list):
        return "", "", payload

    text_parts: list[str] = []
    think_parts: list[str] = []
    for blk in blocks:
        if not isinstance(blk, dict):
            if isinstance(blk, str):
                text_parts.append(blk)
            continue
        btype = blk.get("type")
        if btype == "text":
            text_parts.append(blk.get("text", "") or "")
        elif btype == "thinking":
            think_parts.append(blk.get("thinking", "") or "")
        elif btype is None and isinstance(blk.get("text"), str):
            text_parts.append(blk["text"])
    return "".join(text_parts), "".join(think_parts), payload


async def _generate_claude(a: Args, sample: Sample, claude_cfg: dict) -> Sample:
    """Multi-turn rollout against a Bedrock-style Claude invoke endpoint.

    Images in the initial user message are encoded ONCE (in parallel via
    threads) and reused across turns; only the new text-only assistant/obs
    messages are appended each round.
    """
    import importlib
    import aiohttp

    env_path = getattr(a, "rollout_interaction_env_path", "vdr_core.env")
    env_module = importlib.import_module(env_path)
    env = env_module.build_env(sample=sample, args=a)
    env.reset()

    raw_prompt = sample.metadata.get("raw_prompt", [])
    trajectory: list[dict] = [dict(m) for m in raw_prompt]

    max_turns = getattr(a, "max_turns", 20) or 20
    max_dim = int(claude_cfg.get("max_image_dim", _DEFAULT_CLAUDE_MAX_IMAGE_DIM))
    max_bytes = int(claude_cfg.get("max_image_bytes", _DEFAULT_CLAUDE_MAX_IMAGE_BYTES))
    last_text = ""

    # ----- pre-encode initial user message (heavy work, done once) -----
    system_text, initial_user_raw = _split_claude_system(raw_prompt)
    initial_user_msg = await _build_claude_initial_user(
        initial_user_raw, max_dim, max_bytes
    )
    extra_turns: list[dict] = []  # text-only assistant/user messages appended each turn

    headers = {
        "Content-Type": "application/json",
        "token": claude_cfg["token"],
    }
    timeout = aiohttp.ClientTimeout(total=float(claude_cfg.get("timeout", 600)))

    # Static body bits computed once. Keep the request minimal — match the
    # reference curl body (anthropic_version + max_tokens + messages). Only
    # add system / model / thinking when explicitly needed; do NOT carry over
    # OpenAI-style sampling params (temperature/top_p/top_k).
    base_body: dict = {
        "anthropic_version": claude_cfg.get("anthropic_version", "bedrock-2023-05-31"),
        "max_tokens": int(claude_cfg.get("max_tokens", 8192)),
    }
    if system_text:
        base_body["system"] = system_text
    if claude_cfg.get("model"):
        base_body["model"] = claude_cfg["model"]
    if claude_cfg.get("thinking_enabled"):
        budget = int(claude_cfg.get("thinking_budget", 4000))
        if budget >= base_body["max_tokens"]:
            budget = max(1, base_body["max_tokens"] - 1)
        base_body["thinking"] = {"type": "enabled", "budget_tokens": budget}

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for _turn in range(max_turns):
            body = dict(base_body)
            body["messages"] = [initial_user_msg] + extra_turns

            try:
                t0 = time.time()
                async with session.post(
                    claude_cfg["url"], headers=headers, json=body
                ) as resp:
                    raw_body = await resp.text()
                    if resp.status >= 400:
                        log.error(
                            "Claude HTTP %d at turn %d (%.2fs): %s",
                            resp.status, _turn, time.time() - t0, raw_body[:800],
                        )
                        break
                    try:
                        data = json.loads(raw_body)
                    except json.JSONDecodeError:
                        log.error(
                            "Claude turn %d: non-JSON response (%.2fs): %s",
                            _turn, time.time() - t0, raw_body[:800],
                        )
                        break
            except Exception as exc:
                log.error("Claude call failed at turn %d: %s", _turn, exc)
                break

            assistant_text, thinking_text, payload = _extract_claude_response(data)

            if not assistant_text and not thinking_text:
                preview = (raw_body[:800] + "...") if len(raw_body) > 800 else raw_body
                log.error(
                    "Claude turn %d returned no parseable content. raw=%s",
                    _turn, preview,
                )
                break

            if _turn == 0:
                stop_reason = payload.get("stop_reason") if isinstance(payload, dict) else None
                usage = payload.get("usage") if isinstance(payload, dict) else None
                log.info(
                    "Claude turn 0 ok (%.2fs): %d chars text, %d chars thinking, "
                    "stop=%s usage=%s",
                    time.time() - t0, len(assistant_text), len(thinking_text),
                    stop_reason, usage,
                )

            stored_text = (
                f"<think>{thinking_text}</think>\n{assistant_text}"
                if thinking_text else assistant_text
            )
            last_text = stored_text
            msg_asst_text = {"role": "assistant", "content": stored_text}
            trajectory.append(msg_asst_text)
            extra_turns.append({
                "role": "assistant",
                "content": [{"type": "text", "text": stored_text}],
            })

            if ANSWER_RE.search(assistant_text):
                break

            obs, done, _info = await asyncio.to_thread(env.step, assistant_text)
            obs_str = obs.get("obs_str", "") or ""
            trajectory.append({"role": "user", "content": obs_str})
            extra_turns.append({
                "role": "user",
                "content": [{"type": "text", "text": obs_str}],
            })

            if done:
                break

    sample.response = last_text
    sample.metadata["trajectory"] = trajectory
    return sample


# ---------------------------------------------------------------------------
# Gemini (generateContent) rollout helpers
# ---------------------------------------------------------------------------

_DEFAULT_GEMINI_MAX_IMAGE_BYTES = int(3.6 * 1024 * 1024)
_DEFAULT_GEMINI_MAX_IMAGE_DIM = 8000


def _gemini_encode_image(
    path: str,
    max_dim: int = _DEFAULT_GEMINI_MAX_IMAGE_DIM,
    max_bytes: int = _DEFAULT_GEMINI_MAX_IMAGE_BYTES,
) -> tuple[str, str]:
    """JPEG-encode + base64 an image; reuse Claude's encoder (same constraints)."""
    return _claude_encode_image(path, max_dim=max_dim, max_bytes=max_bytes)


async def _build_gemini_initial_user(
    user_msg: dict | None, max_dim: int, max_bytes: int
) -> dict:
    """First `content` entry for Gemini. Images encoded in parallel."""
    if user_msg is None:
        return {"role": "user", "parts": [{"text": ""}]}
    content = user_msg.get("content")
    if isinstance(content, str):
        return {"role": "user", "parts": [{"text": content}]}

    parts: list[Any] = []
    encode_tasks: list = []
    encode_slots: list[int] = []
    for item in content or []:
        if not isinstance(item, dict):
            continue
        t = item.get("type")
        if t == "text":
            parts.append({"text": item.get("text", "")})
        elif t == "image":
            path = item.get("image", "")
            if path and os.path.exists(path):
                encode_slots.append(len(parts))
                parts.append(None)
                encode_tasks.append(
                    asyncio.to_thread(_gemini_encode_image, path, max_dim, max_bytes)
                )

    if encode_tasks:
        t0 = time.time()
        encoded = await asyncio.gather(*encode_tasks)
        log.info("Gemini image encode: %d images in %.2fs", len(encode_tasks), time.time() - t0)
        for pos, (media_type, b64) in zip(encode_slots, encoded):
            parts[pos] = {"inlineData": {"mimeType": media_type, "data": b64}}

    return {"role": "user", "parts": parts or [{"text": ""}]}


def _extract_gemini_response(data: Any) -> tuple[str, str, dict | None]:
    """Return (text, thinking, normalized_payload) from a Gemini response.

    Handles raw generateContent shape and proxies that wrap the body inside
    `body`/`data`/`result`/`response`/`output`/`message`.
    """
    payload = data
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            return "", "", None
    if not isinstance(payload, dict):
        return "", "", None

    if "candidates" not in payload:
        for key in ("body", "data", "result", "response", "output", "message"):
            sub = payload.get(key)
            if isinstance(sub, str):
                try:
                    sub = json.loads(sub)
                except Exception:
                    continue
            if isinstance(sub, dict) and "candidates" in sub:
                payload = sub
                break

    cands = payload.get("candidates")
    if not isinstance(cands, list) or not cands:
        return "", "", payload
    first = cands[0]
    if not isinstance(first, dict):
        return "", "", payload
    content = first.get("content") or {}
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list):
        return "", "", payload

    text_parts: list[str] = []
    think_parts: list[str] = []
    for blk in parts:
        if not isinstance(blk, dict):
            if isinstance(blk, str):
                text_parts.append(blk)
            continue
        txt = blk.get("text", "") or ""
        if not txt:
            continue
        if blk.get("thought") is True:
            think_parts.append(txt)
        else:
            text_parts.append(txt)
    return "".join(text_parts), "".join(think_parts), payload


async def _generate_gemini(a: Args, sample: Sample, gemini_cfg: dict) -> Sample:
    """Multi-turn rollout against Gemini generateContent.

    Initial-user images are encoded ONCE (parallel via threads); subsequent
    turns only append text-only model/user parts.
    """
    import importlib
    import aiohttp

    env_path = getattr(a, "rollout_interaction_env_path", "vdr_core.env")
    env_module = importlib.import_module(env_path)
    env = env_module.build_env(sample=sample, args=a)
    env.reset()

    raw_prompt = sample.metadata.get("raw_prompt", [])
    trajectory: list[dict] = [dict(m) for m in raw_prompt]

    max_turns = getattr(a, "max_turns", 20) or 20
    max_dim = int(gemini_cfg.get("max_image_dim", _DEFAULT_GEMINI_MAX_IMAGE_DIM))
    max_bytes = int(gemini_cfg.get("max_image_bytes", _DEFAULT_GEMINI_MAX_IMAGE_BYTES))
    last_text = ""

    system_text, initial_user_raw = _split_claude_system(raw_prompt)
    initial_user_msg = await _build_gemini_initial_user(
        initial_user_raw, max_dim, max_bytes
    )
    extra_turns: list[dict] = []

    headers = {
        "Content-Type": "application/json",
        "api-key": gemini_cfg["api_key"],
    }
    timeout = aiohttp.ClientTimeout(total=float(gemini_cfg.get("timeout", 600)))

    gen_cfg: dict = {
        "maxOutputTokens": int(gemini_cfg.get("max_tokens", 8192)),
    }
    if gemini_cfg.get("temperature") is not None:
        gen_cfg["temperature"] = float(gemini_cfg["temperature"])
    if gemini_cfg.get("top_p") is not None:
        gen_cfg["topP"] = float(gemini_cfg["top_p"])

    thinking_cfg: dict = {}
    if gemini_cfg.get("thinking_level"):
        thinking_cfg["thinkingLevel"] = str(gemini_cfg["thinking_level"]).upper()
    elif gemini_cfg.get("thinking_budget") is not None:
        thinking_cfg["thinkingBudget"] = int(gemini_cfg["thinking_budget"])
    if gemini_cfg.get("include_thoughts"):
        thinking_cfg["includeThoughts"] = True
    if thinking_cfg:
        gen_cfg["thinkingConfig"] = thinking_cfg

    base_body: dict = {"generationConfig": gen_cfg}
    if system_text:
        base_body["systemInstruction"] = {"parts": [{"text": system_text}]}
    if gemini_cfg.get("model"):
        base_body["model"] = gemini_cfg["model"]

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for _turn in range(max_turns):
            body = dict(base_body)
            body["contents"] = [initial_user_msg] + extra_turns

            try:
                t0 = time.time()
                async with session.post(
                    gemini_cfg["url"], headers=headers, json=body
                ) as resp:
                    raw_body = await resp.text()
                    if resp.status >= 400:
                        log.error(
                            "Gemini HTTP %d at turn %d (%.2fs): %s",
                            resp.status, _turn, time.time() - t0, raw_body[:800],
                        )
                        break
                    try:
                        data = json.loads(raw_body)
                    except json.JSONDecodeError:
                        log.error(
                            "Gemini turn %d: non-JSON response (%.2fs): %s",
                            _turn, time.time() - t0, raw_body[:800],
                        )
                        break
            except Exception as exc:
                log.error("Gemini call failed at turn %d: %s", _turn, exc)
                break

            assistant_text, thinking_text, payload = _extract_gemini_response(data)

            if not assistant_text and not thinking_text:
                preview = (raw_body[:800] + "...") if len(raw_body) > 800 else raw_body
                log.error(
                    "Gemini turn %d returned no parseable content. raw=%s",
                    _turn, preview,
                )
                break

            if _turn == 0:
                cand0 = (payload.get("candidates") or [{}])[0] if isinstance(payload, dict) else {}
                finish = cand0.get("finishReason") if isinstance(cand0, dict) else None
                usage = payload.get("usageMetadata") if isinstance(payload, dict) else None
                log.info(
                    "Gemini turn 0 ok (%.2fs): %d chars text, %d chars thinking, "
                    "finish=%s usage=%s",
                    time.time() - t0, len(assistant_text), len(thinking_text),
                    finish, usage,
                )

            stored_text = (
                f"<think>{thinking_text}</think>\n{assistant_text}"
                if thinking_text else assistant_text
            )
            last_text = stored_text
            trajectory.append({"role": "assistant", "content": stored_text})
            extra_turns.append({
                "role": "model",
                "parts": [{"text": stored_text}],
            })

            if ANSWER_RE.search(assistant_text):
                break

            obs, done, _info = await asyncio.to_thread(env.step, assistant_text)
            obs_str = obs.get("obs_str", "") or ""
            trajectory.append({"role": "user", "content": obs_str})
            extra_turns.append({
                "role": "user",
                "parts": [{"text": obs_str}],
            })

            if done:
                break

    sample.response = last_text
    sample.metadata["trajectory"] = trajectory
    return sample


# ---------------------------------------------------------------------------
# GPT (Azure-style chat completions) rollout helpers
# ---------------------------------------------------------------------------

_DEFAULT_GPT_MAX_IMAGE_BYTES = int(3.6 * 1024 * 1024)
_DEFAULT_GPT_MAX_IMAGE_DIM = 8000


def _gpt_encode_image(
    path: str,
    max_dim: int = _DEFAULT_GPT_MAX_IMAGE_DIM,
    max_bytes: int = _DEFAULT_GPT_MAX_IMAGE_BYTES,
) -> tuple[str, str]:
    """JPEG-encode + base64 an image; reuse Claude's encoder (same constraints)."""
    return _claude_encode_image(path, max_dim=max_dim, max_bytes=max_bytes)


async def _build_gpt_initial_user(
    user_msg: dict | None, max_dim: int, max_bytes: int
) -> dict:
    """First user message for GPT chat completions; images encoded in parallel."""
    if user_msg is None:
        return {"role": "user", "content": [{"type": "text", "text": ""}]}
    content = user_msg.get("content")
    if isinstance(content, str):
        return {"role": "user", "content": [{"type": "text", "text": content}]}

    parts: list[Any] = []
    encode_tasks: list = []
    encode_slots: list[int] = []
    for item in content or []:
        if not isinstance(item, dict):
            continue
        t = item.get("type")
        if t == "text":
            parts.append({"type": "text", "text": item.get("text", "")})
        elif t == "image":
            path = item.get("image", "")
            if path and os.path.exists(path):
                encode_slots.append(len(parts))
                parts.append(None)
                encode_tasks.append(
                    asyncio.to_thread(_gpt_encode_image, path, max_dim, max_bytes)
                )

    if encode_tasks:
        t0 = time.time()
        encoded = await asyncio.gather(*encode_tasks)
        log.info("GPT image encode: %d images in %.2fs", len(encode_tasks), time.time() - t0)
        for pos, (media_type, b64) in zip(encode_slots, encoded):
            parts[pos] = {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{b64}"},
            }

    return {"role": "user", "content": parts or [{"type": "text", "text": ""}]}


def _extract_gpt_response(data: Any) -> tuple[str, str, dict | None]:
    """Return (text, thinking, normalized_payload) from a chat-completions response.

    Tolerates one envelope layer (body/data/result/response/output/message) like
    the Claude/Gemini extractors. Pulls `reasoning_content` / `reasoning` if the
    proxy attaches a thinking trace.
    """
    payload = data
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            return "", "", None
    if not isinstance(payload, dict):
        return "", "", None

    if "choices" not in payload:
        for key in ("body", "data", "result", "response", "output", "message"):
            sub = payload.get(key)
            if isinstance(sub, str):
                try:
                    sub = json.loads(sub)
                except Exception:
                    continue
            if isinstance(sub, dict) and "choices" in sub:
                payload = sub
                break

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return "", "", payload
    first = choices[0]
    if not isinstance(first, dict):
        return "", "", payload
    msg = first.get("message")
    if not isinstance(msg, dict):
        return "", "", payload

    text = ""
    content = msg.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        chunks: list[str] = []
        for it in content:
            if isinstance(it, dict) and it.get("type") in ("text", None):
                chunks.append(it.get("text", "") or "")
            elif isinstance(it, str):
                chunks.append(it)
        text = "".join(chunks)

    thinking = ""
    raw_think = msg.get("reasoning_content") or msg.get("reasoning")
    if isinstance(raw_think, str):
        thinking = raw_think
    elif isinstance(raw_think, list):
        chunks: list[str] = []
        for it in raw_think:
            if isinstance(it, dict):
                chunks.append(it.get("text", "") or "")
            elif isinstance(it, str):
                chunks.append(it)
        thinking = "".join(chunks)

    return text, thinking, payload


async def _generate_gpt(a: Args, sample: Sample, gpt_cfg: dict) -> Sample:
    """Multi-turn rollout against an Azure-style chat-completions endpoint.

    Initial-user images are encoded ONCE (parallel via threads); subsequent
    turns only append text-only assistant/user messages.
    """
    import importlib
    import aiohttp

    env_path = getattr(a, "rollout_interaction_env_path", "vdr_core.env")
    env_module = importlib.import_module(env_path)
    env = env_module.build_env(sample=sample, args=a)
    env.reset()

    raw_prompt = sample.metadata.get("raw_prompt", [])
    trajectory: list[dict] = [dict(m) for m in raw_prompt]

    max_turns = getattr(a, "max_turns", 20) or 20
    max_dim = int(gpt_cfg.get("max_image_dim", _DEFAULT_GPT_MAX_IMAGE_DIM))
    max_bytes = int(gpt_cfg.get("max_image_bytes", _DEFAULT_GPT_MAX_IMAGE_BYTES))
    last_text = ""

    system_text, initial_user_raw = _split_claude_system(raw_prompt)
    initial_user_msg = await _build_gpt_initial_user(
        initial_user_raw, max_dim, max_bytes
    )
    sys_msgs: list[dict] = (
        [{"role": "system", "content": system_text}] if system_text else []
    )
    extra_turns: list[dict] = []

    headers = {
        "Content-Type": "application/json",
        "api-key": gpt_cfg["api_key"],
    }
    timeout = aiohttp.ClientTimeout(total=float(gpt_cfg.get("timeout", 600)))

    # GPT-5 / o-series Azure deployments require `max_completion_tokens` and
    # reject the legacy `max_tokens`. They also tend to reject custom
    # temperature/top_p (only the default value is accepted), so only send
    # those when the user explicitly opts in via config.
    base_body: dict = {
        "max_completion_tokens": int(gpt_cfg.get("max_tokens", 8192)),
    }
    if gpt_cfg.get("model"):
        base_body["model"] = gpt_cfg["model"]
    if gpt_cfg.get("temperature") is not None:
        base_body["temperature"] = float(gpt_cfg["temperature"])
    if gpt_cfg.get("top_p") is not None:
        base_body["top_p"] = float(gpt_cfg["top_p"])

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for _turn in range(max_turns):
            body = dict(base_body)
            body["messages"] = sys_msgs + [initial_user_msg] + extra_turns

            try:
                t0 = time.time()
                async with session.post(
                    gpt_cfg["url"], headers=headers, json=body
                ) as resp:
                    raw_body = await resp.text()
                    if resp.status >= 400:
                        log.error(
                            "GPT HTTP %d at turn %d (%.2fs): %s",
                            resp.status, _turn, time.time() - t0, raw_body[:800],
                        )
                        break
                    try:
                        data = json.loads(raw_body)
                    except json.JSONDecodeError:
                        log.error(
                            "GPT turn %d: non-JSON response (%.2fs): %s",
                            _turn, time.time() - t0, raw_body[:800],
                        )
                        break
            except Exception as exc:
                log.error("GPT call failed at turn %d: %s", _turn, exc)
                break

            assistant_text, thinking_text, payload = _extract_gpt_response(data)

            if not assistant_text and not thinking_text:
                preview = (raw_body[:800] + "...") if len(raw_body) > 800 else raw_body
                log.error(
                    "GPT turn %d returned no parseable content. raw=%s",
                    _turn, preview,
                )
                break

            if _turn == 0:
                ch0 = (payload.get("choices") or [{}])[0] if isinstance(payload, dict) else {}
                finish = ch0.get("finish_reason") if isinstance(ch0, dict) else None
                usage = payload.get("usage") if isinstance(payload, dict) else None
                log.info(
                    "GPT turn 0 ok (%.2fs): %d chars text, %d chars thinking, "
                    "finish=%s usage=%s",
                    time.time() - t0, len(assistant_text), len(thinking_text),
                    finish, usage,
                )

            stored_text = (
                f"<think>{thinking_text}</think>\n{assistant_text}"
                if thinking_text else assistant_text
            )
            last_text = stored_text
            trajectory.append({"role": "assistant", "content": stored_text})
            extra_turns.append({"role": "assistant", "content": stored_text})

            if ANSWER_RE.search(assistant_text):
                break

            obs, done, _info = await asyncio.to_thread(env.step, assistant_text)
            obs_str = obs.get("obs_str", "") or ""
            trajectory.append({"role": "user", "content": obs_str})
            extra_turns.append({"role": "user", "content": obs_str})

            if done:
                break

    sample.response = last_text
    sample.metadata["trajectory"] = trajectory
    return sample


# ---------------------------------------------------------------------------
# Per-sample async rollout
# ---------------------------------------------------------------------------

async def run_one(a, s, idx, params, sem, timeout, reward_fn, extra_meta,
                  openai_cfg=None, claude_cfg=None, gemini_cfg=None, gpt_cfg=None, direct=False):
    async with sem:
        x, r, e, e_str = clone(s, idx), None, None, None
        try:
            if claude_cfg:
                coro = _generate_claude(a, x, claude_cfg)
            elif gemini_cfg:
                coro = _generate_gemini(a, x, gemini_cfg)
            elif gpt_cfg:
                coro = _generate_gpt(a, x, gpt_cfg)
            elif openai_cfg:
                coro = _generate_openai(a, x, openai_cfg)
            else:
                coro = generate(args=a, sample=x, sampling_params=dict(params), evaluation=True)
            r = await asyncio.wait_for(coro, timeout=timeout) if timeout else await coro
        except Exception as exc:
            e = exc
            log.exception("sample %s failed", idx)
            import traceback
            e_str = traceback.format_exc()

        src = r or x
        rollout_ok = r is not None and e is None

        tr = _json_safe(src.metadata.get("trajectory") or src.metadata.get("raw_prompt") or [])
        tr = clean_trajectory(tr if isinstance(tr, list) else [])
        response = clean_end_tokens(src.response)
        question = extract_first_user_question(tr, src.prompt)

        if direct:
            # Use last assistant message from trajectory (correct text from SGLang),
            # not src.response which is decoded from token IDs via hf_checkpoint tokenizer
            # and would be garbled when the eval model differs from hf_checkpoint.
            reward_response = extract_last_assistant_response(tr, response)
            aa = reward_response
        else:
            aa = ans_traj(tr) or ans(response)
            reward_response = extract_last_assistant_response(tr, response)

        reward_call_ok, is_correct = True, None
        if reward_fn is not None:
            try:
                reward_result = await reward_fn.async_score(
                    question=question, response=reward_response, answer=src.label
                )
                is_correct = reward_result.get("is_correct")
            except Exception:
                reward_call_ok = False

        return {
            "index":          idx,
            "video_id":       extra_meta.get("video_id", ""),
            "category":       extra_meta.get("category", ""),
            "difficulty":     extra_meta.get("difficulty", ""),
            "rollout_ok":     rollout_ok,
            "reward_call_ok": reward_call_ok,
            "is_correct":     is_correct,
            "answer":         clean_end_tokens(aa),
            "label":          None if src.label is None else str(src.label),
            "num_turns":      turns(tr),
            "ok":             rollout_ok and reward_call_ok,
            "exception":      e_str,
            "trajectory":     add_think_prefix(tr),
        }


async def close_session():
    import vdr_core.rollout as ro
    s = getattr(ro, "_HTTP_SESSION", None)
    if s is not None and not s.closed:
        await s.close()
    ro._HTTP_SESSION = None


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

def compute_summary(results: list[dict]) -> dict:
    total = scored = correct = 0
    by_cat: dict[str, list] = defaultdict(list)
    by_diff: dict[str, list] = defaultdict(list)

    for r in results:
        total += 1
        ic = r.get("is_correct")
        cat = r.get("category", "unknown")
        diff = r.get("difficulty", "unknown")
        if ic is not None:
            scored += 1
            v = 1 if ic else 0
            correct += v
            by_cat[cat].append(v)
            by_diff[diff].append(v)

    def acc(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    return {
        "total":    total,
        "scored":   scored,
        "accuracy": acc([1 if r.get("is_correct") else 0 for r in results if r.get("is_correct") is not None]),
        "by_category":  {k: {"n": len(v), "accuracy": acc(v)} for k, v in sorted(by_cat.items())},
        "by_difficulty": {k: {"n": len(v), "accuracy": acc(v)} for k, v in sorted(by_diff.items())},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(c):
    cfg = yaml.safe_load(open(c.config, encoding="utf-8")) if c.config else {}
    cfg = cfg or {}
    ip, port = parse_url(c.sglang_url)
    cfg.update({
        "hf_checkpoint":       c.hf_checkpoint,
        "sglang_router_ip":    ip,
        "sglang_router_port":  port,
        "max_turns":           c.max_turns or cfg.get("max_turns", 20),
        "rollout_max_response_len": c.max_new_tokens or cfg.get("rollout_max_response_len", 32000),
        "rollout_max_context_len": (
            c.rollout_max_context_len
            if c.rollout_max_context_len is not None
            else cfg.get("rollout_max_context_len")
        ),
        "rollout_temperature": (
            c.temperature if c.temperature is not None else cfg.get("rollout_temperature", 0.0)
        ),
        "rollout_top_p": c.top_p if c.top_p is not None else cfg.get("rollout_top_p", 0.9),
        "rollout_top_k": c.top_k if c.top_k is not None else cfg.get("rollout_top_k", 20),
    })

    # Mode-specific setup
    mode = getattr(c, "mode", "tool")
    eval_dir = Path(c.config).parent

    if mode == "direct":
        prompt_file = c.system_prompt_file or str(eval_dir / "direct_system_prompt.txt")
        template = Path(prompt_file).read_text(encoding="utf-8")
        system_content = _build_direct_system_content(template)
        build_messages_fn = _build_messages_direct
        log.info("Mode: direct (single-turn, no tools)")
    else:
        prompt_file = c.system_prompt_file or str(eval_dir / "eval_system_prompt.txt")
        template = Path(prompt_file).read_text(encoding="utf-8")
        supported_tool_names = [
            t.strip()
            for t in str(cfg.get("supported_tools", "search,visit,select_crop_search")).split(",")
            if t.strip()
        ]
        system_content = _build_system_content(template, supported_tool_names)
        build_messages_fn = _build_messages
        log.info("Mode: tool (multi-turn, tools=%s)", supported_tool_names)

    if c.max_turns:
        cfg["max_turns"] = c.max_turns
    # Direct mode is always single-turn — must override AFTER CLI arg processing
    if mode == "direct":
        cfg["max_turns"] = 1

    a = Args(cfg)
    tok = load_tokenizer(a.hf_checkpoint, trust_remote_code=True)
    proc = load_processor(a.hf_checkpoint, trust_remote_code=True)

    # Load input (jsonl / csv). Format is auto-detected from extension unless
    # the user passes --jsonl explicitly. JSONL is the default when neither
    # --csv nor --jsonl is set.
    jsonl_path = getattr(c, "jsonl", None)
    csv_path = getattr(c, "csv", None)
    fmt = getattr(c, "format", "auto") or "auto"
    if fmt == "auto":
        if jsonl_path:
            fmt = "jsonl"
        elif csv_path:
            ext = Path(csv_path).suffix.lower()
            fmt = "jsonl" if ext in (".jsonl", ".ndjson", ".json") else "csv"
        else:
            raise ValueError("Pass --jsonl or --csv")
    input_path = jsonl_path if fmt == "jsonl" else csv_path
    if not input_path:
        raise ValueError(f"Format is {fmt!r} but the matching path arg is unset")

    if fmt == "jsonl":
        records = load_jsonl(input_path)
    else:
        records = load_csv(input_path)
    if c.video_ids:
        keep = set(c.video_ids.split(","))
        records = [r for r in records if r["video_id"] in keep]
    log.info("Loaded %d records from %s (%s)", len(records), input_path, fmt)

    # Build samples
    all_entries: list[tuple[dict, Sample]] = []
    missing = []
    for rec in records:
        # Per-row `images` (JSONL) wins; otherwise resolve via frames_dir/video_id.
        frames = rec.get("images") or find_frames(c.frames_dir, rec["video_id"])
        if not frames:
            missing.append(rec["video_id"])
            log.warning("No frames found for video %s — skipping", rec["video_id"])
            continue
        s = _build_sample(
            image_paths=frames,
            question=rec["question"],
            label=rec["answer"],
            system_content=system_content,
            metadata={"video_id": rec["video_id"], "category": rec["category"], "difficulty": rec["difficulty"]},
            tokenizer=tok,
            processor=proc,
            build_messages_fn=build_messages_fn,
        )
        all_entries.append((rec, s))

    if missing:
        log.warning("%d videos skipped (no frames): %s", len(missing), missing)

    entries = all_entries[c.start_index:]
    if c.limit is not None:
        entries = entries[: c.limit]

    # Output paths — separate subdir per mode so runs don't overwrite each other
    model_name = getattr(c, "model_name", None) or Path(c.hf_checkpoint).name
    output_dir = Path(c.output_dir) / model_name / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    traj_path = output_dir / "trajectories.jsonl"
    summary_path = output_dir / "summary.json"

    existing = load_existing_indices(str(traj_path))
    indexed = [
        (c.start_index + i, rec, s)
        for i, (rec, s) in enumerate(entries)
        if c.start_index + i not in existing
    ]
    skipped = len(entries) - len(indexed)
    if skipped:
        log.info("skipping %d already-processed samples", skipped)

    sem = asyncio.Semaphore(c.max_async_samples)
    params = sampling_params(a)

    # Build backend-specific configs
    backend = getattr(c, "backend", "sglang")
    openai_cfg: dict | None = None
    claude_cfg: dict | None = None
    gemini_cfg: dict | None = None
    gpt_cfg: dict | None = None
    if backend == "gpt":
        gpt_url = getattr(c, "gpt_url", None) or cfg.get("gpt_url", "")
        gpt_api_key = (
            getattr(c, "gpt_api_key", None)
            or cfg.get("gpt_api_key", "")
            or os.environ.get("GPT_API_KEY", "")
        )
        if not gpt_url:
            raise ValueError("--gpt-url is required when --backend=gpt")
        if not gpt_api_key:
            raise ValueError("--gpt-api-key (or env GPT_API_KEY) is required when --backend=gpt")
        gpt_cfg = {
            "url": gpt_url,
            "api_key": gpt_api_key,
            "model": getattr(c, "gpt_model", None) or cfg.get("gpt_model"),
            "max_tokens": c.max_new_tokens or cfg.get("rollout_max_response_len", 8192),
            # GPT-5 / o-series only accept the default temperature/top_p, so
            # only forward these when the user *explicitly* passes them on
            # the CLI — never inherit from config.yaml.
            "temperature": c.temperature,
            "top_p": c.top_p,
            "max_image_dim": int(
                getattr(c, "gpt_max_image_dim", None)
                or cfg.get("gpt_max_image_dim", _DEFAULT_GPT_MAX_IMAGE_DIM)
            ),
            "max_image_bytes": int(
                getattr(c, "gpt_max_image_bytes", None)
                or cfg.get("gpt_max_image_bytes", _DEFAULT_GPT_MAX_IMAGE_BYTES)
            ),
            "timeout": float(
                getattr(c, "gpt_timeout", None) or cfg.get("gpt_timeout", 600)
            ),
        }
        log.info(
            "Backend: gpt  url=%s  model=%s", gpt_cfg["url"], gpt_cfg["model"],
        )
    elif backend == "gemini":
        gemini_url = getattr(c, "gemini_url", None) or cfg.get("gemini_url", "")
        gemini_api_key = (
            getattr(c, "gemini_api_key", None)
            or cfg.get("gemini_api_key", "")
            or os.environ.get("GEMINI_API_KEY", "")
        )
        if not gemini_url:
            raise ValueError("--gemini-url is required when --backend=gemini")
        if not gemini_api_key:
            raise ValueError("--gemini-api-key (or env GEMINI_API_KEY) is required when --backend=gemini")
        gemini_cfg = {
            "url": gemini_url,
            "api_key": gemini_api_key,
            "model": getattr(c, "gemini_model", None) or cfg.get("gemini_model"),
            "max_tokens": c.max_new_tokens or cfg.get("rollout_max_response_len", 8192),
            "temperature": (
                c.temperature if c.temperature is not None
                else cfg.get("rollout_temperature")
            ),
            "top_p": c.top_p if c.top_p is not None else cfg.get("rollout_top_p"),
            "thinking_level": (
                getattr(c, "gemini_thinking_level", None)
                or cfg.get("gemini_thinking_level")
            ),
            "thinking_budget": (
                getattr(c, "gemini_thinking_budget", None)
                if getattr(c, "gemini_thinking_budget", None) is not None
                else cfg.get("gemini_thinking_budget")
            ),
            "include_thoughts": bool(getattr(c, "gemini_include_thoughts", False)),
            "max_image_dim": int(
                getattr(c, "gemini_max_image_dim", None)
                or cfg.get("gemini_max_image_dim", _DEFAULT_GEMINI_MAX_IMAGE_DIM)
            ),
            "max_image_bytes": int(
                getattr(c, "gemini_max_image_bytes", None)
                or cfg.get("gemini_max_image_bytes", _DEFAULT_GEMINI_MAX_IMAGE_BYTES)
            ),
            "timeout": float(
                getattr(c, "gemini_timeout", None) or cfg.get("gemini_timeout", 600)
            ),
        }
        log.info(
            "Backend: gemini  url=%s  model=%s  thinking_level=%s  thinking_budget=%s",
            gemini_cfg["url"], gemini_cfg["model"],
            gemini_cfg["thinking_level"], gemini_cfg["thinking_budget"],
        )
    elif backend == "claude":
        claude_url = getattr(c, "claude_url", None) or cfg.get("claude_url", "")
        claude_token = (
            getattr(c, "claude_token", None)
            or cfg.get("claude_token", "")
            or os.environ.get("CLAUDE_TOKEN", "")
        )
        if not claude_url:
            raise ValueError("--claude-url is required when --backend=claude")
        if not claude_token:
            raise ValueError("--claude-token (or env CLAUDE_TOKEN) is required when --backend=claude")
        claude_cfg = {
            "url": claude_url,
            "token": claude_token,
            "model": getattr(c, "claude_model", None) or cfg.get("claude_model"),
            "anthropic_version": getattr(c, "claude_anthropic_version", None)
                or cfg.get("claude_anthropic_version", "bedrock-2023-05-31"),
            "max_tokens": c.max_new_tokens or cfg.get("rollout_max_response_len", 8192),
            "temperature": (
                c.temperature if c.temperature is not None
                else cfg.get("rollout_temperature", 0.0)
            ),
            "top_p": c.top_p if c.top_p is not None else cfg.get("rollout_top_p"),
            "top_k": c.top_k if c.top_k is not None else cfg.get("rollout_top_k"),
            "thinking_enabled": bool(getattr(c, "claude_thinking", False)),
            "thinking_budget": int(
                getattr(c, "claude_thinking_budget", None)
                or cfg.get("claude_thinking_budget", 4000)
            ),
            "max_image_dim": int(
                getattr(c, "claude_max_image_dim", None)
                or cfg.get("claude_max_image_dim", _DEFAULT_CLAUDE_MAX_IMAGE_DIM)
            ),
            "max_image_bytes": int(
                getattr(c, "claude_max_image_bytes", None)
                or cfg.get("claude_max_image_bytes", _DEFAULT_CLAUDE_MAX_IMAGE_BYTES)
            ),
            "timeout": float(
                getattr(c, "claude_timeout", None) or cfg.get("claude_timeout", 600)
            ),
        }
        log.info(
            "Backend: claude  url=%s  model=%s  version=%s  thinking=%s",
            claude_cfg["url"], claude_cfg["model"], claude_cfg["anthropic_version"],
            claude_cfg["thinking_enabled"],
        )
    elif backend in ("openai", "vllm"):
        raw_headers = cfg.get("openai_headers") or {}
        if isinstance(raw_headers, str):
            raw_headers = json.loads(raw_headers)
        if backend == "vllm":
            vllm_url = getattr(c, "vllm_url", None) or cfg.get("vllm_url", "http://localhost:8000/v1")
            vllm_model = getattr(c, "vllm_model", None) or cfg.get("vllm_model", "")
            openai_cfg = {
                "api_key":     "EMPTY",
                "base_url":    vllm_url,
                "model":       vllm_model,
                "headers":     {},
                "max_tokens":  c.max_new_tokens or cfg.get("rollout_max_response_len", 8192),
                "temperature": c.temperature if c.temperature is not None else cfg.get("rollout_temperature", 0.0),
            }
            log.info("Backend: vllm  model=%s  base_url=%s", openai_cfg["model"], openai_cfg["base_url"])
        else:
            openai_cfg = {
                "api_key":     getattr(c, "openai_api_key", None) or cfg.get("openai_api_key", ""),
                "base_url":    getattr(c, "openai_base_url", None) or cfg.get("openai_base_url", ""),
                "model":       getattr(c, "openai_model", None) or cfg.get("openai_model", ""),
                "headers":     raw_headers,
                "max_tokens":  c.max_new_tokens or cfg.get("rollout_max_response_len", 8192),
                "temperature": c.temperature if c.temperature is not None else cfg.get("rollout_temperature", 0.0),
            }
            log.info("Backend: openai  model=%s  base_url=%s", openai_cfg["model"], openai_cfg["base_url"])

    # For reward, vllm is treated as openai (both use chat completions).
    # Resolution priority for vllm/openai reward (base_url / model / api_key):
    #   1. reward-specific flag      (--reward-model-url / --reward-model / --reward-api-key)
    #   2. rollout-backend flag      (--vllm-url|--openai-base-url / --vllm-model|--openai-model)
    #   3. config.yaml               (vllm_url / openai_base_url, vllm_model / openai_model)
    # so a remote vLLM judge can be wired in without touching the rollout endpoint.
    reward_backend = getattr(c, "reward_backend", None)
    if reward_backend is None:
        # Claude rollout has no claude-reward impl — fall back to sglang.
        reward_backend = backend if backend in ("sglang", "openai", "vllm") else "sglang"
    is_openai_like_reward = reward_backend in ("openai", "vllm")
    if reward_backend == "vllm":
        reward_backend = "openai"

    reward_url_arg = getattr(c, "reward_model_url", None)
    reward_model_arg = getattr(c, "reward_model", None)
    reward_api_key_arg = getattr(c, "reward_api_key", None)

    if is_openai_like_reward:
        fallback_base_url = (
            (getattr(c, "vllm_url", None) or cfg.get("vllm_url", "http://localhost:8000/v1"))
            if backend == "vllm"
            else (getattr(c, "openai_base_url", None) or cfg.get("openai_base_url"))
        )
        fallback_model = (
            (getattr(c, "vllm_model", None) or cfg.get("vllm_model"))
            if backend == "vllm"
            else (getattr(c, "openai_model", None) or cfg.get("openai_model"))
        )
        reward_openai_base_url = reward_url_arg or fallback_base_url
        reward_openai_model = reward_model_arg or fallback_model
        reward_openai_api_key = (
            reward_api_key_arg
            or getattr(c, "openai_api_key", None)
            or cfg.get("openai_api_key")
            or "EMPTY"
        )
        # sglang reads `url` directly; openai/vllm read `openai_base_url`.
        # If reward is openai-like we don't want `url` to also be the same value
        # (it would override `judge_url` and confuse env-var fallback logic).
        reward_url_for_sglang: str | None = None
        log.info(
            "Reward backend: %s  model=%s  base_url=%s",
            "vllm" if reward_openai_base_url and "/v1" in reward_openai_base_url else "openai",
            reward_openai_model, reward_openai_base_url,
        )
    else:
        reward_openai_base_url = (
            getattr(c, "openai_base_url", None) or cfg.get("openai_base_url")
        )
        reward_openai_model = (
            getattr(c, "openai_model", None) or cfg.get("openai_model")
        )
        reward_openai_api_key = (
            getattr(c, "openai_api_key", None) or cfg.get("openai_api_key") or "EMPTY"
        )
        reward_url_for_sglang = reward_url_arg
        log.info("Reward backend: sglang  url=%s", reward_url_for_sglang)

    if getattr(c, "skip_reward", False):
        reward_fn = None
        log.info("Reward: SKIPPED (--skip-reward); will only run trajectories")
    else:
        reward_fn = DeepResearchReward(
            url=reward_url_for_sglang,
            timeout=c.reward_timeout,
            retry_attempts=c.reward_retry_attempts,
            retry_delay=c.reward_retry_delay,
            temperature=c.reward_temperature,
            max_new_tokens=c.reward_max_new_tokens,
            backend=reward_backend,
            openai_api_key=reward_openai_api_key,
            openai_base_url=reward_openai_base_url,
            openai_model=reward_openai_model,
            openai_headers=cfg.get("openai_headers") or {},
        )

    tasks = [
        asyncio.create_task(
            run_one(
                a, s, idx, params, sem, c.sample_timeout, reward_fn,
                {"video_id": rec["video_id"], "category": rec["category"], "difficulty": rec["difficulty"]},
                openai_cfg=openai_cfg,
                claude_cfg=claude_cfg,
                gemini_cfg=gemini_cfg,
                gpt_cfg=gpt_cfg,
                direct=(mode == "direct"),
            )
        )
        for idx, rec, s in indexed
    ]

    all_results: list[dict] = []

    try:
        with (
            open(str(traj_path), "a", encoding="utf-8") as traj_f,
            tqdm(total=len(entries), desc="Evaluating", unit="video", initial=skipped) as pbar,
        ):
            for fut in asyncio.as_completed(tasks):
                rec_out = await fut
                row = {
                    "index":          rec_out["index"],
                    "video_id":       rec_out["video_id"],
                    "category":       rec_out["category"],
                    "difficulty":     rec_out["difficulty"],
                    "rollout_ok":     rec_out["rollout_ok"],
                    "reward_call_ok": rec_out["reward_call_ok"],
                    "is_correct":     rec_out["is_correct"],
                    "answer":         rec_out["answer"],
                    "label":          rec_out["label"],
                    "num_turns":      rec_out["num_turns"],
                    "ok":             rec_out["ok"],
                    "trajectory":     rec_out["trajectory"],
                }
                traj_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                traj_f.flush()
                all_results.append(rec_out)
                pbar.update(1)
                pbar.set_postfix(
                    correct=rec_out["is_correct"],
                    vid=rec_out["video_id"],
                )
                log.info(
                    "video=%s  is_correct=%s  rollout_ok=%s",
                    rec_out["video_id"], rec_out["is_correct"], rec_out["rollout_ok"],
                )
    finally:
        await close_session()

    # Also collect already-existing results for summary
    existing_results = []
    if existing:
        with open(str(traj_path), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if r.get("index") in existing:
                        existing_results.append(r)
                except json.JSONDecodeError:
                    pass
    combined = existing_results + all_results

    summary = compute_summary(combined)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n========== Evaluation Summary ==========")
    print(f"  Total:    {summary['total']}")
    print(f"  Scored:   {summary['scored']}")
    print(f"  Accuracy: {summary['accuracy']}")
    print("\n  By category:")
    for cat, v in summary["by_category"].items():
        print(f"    {cat:<20} n={v['n']}  acc={v['accuracy']}")
    print("\n  By difficulty:")
    for diff, v in summary["by_difficulty"].items():
        print(f"    {diff:<10} n={v['n']}  acc={v['accuracy']}")
    print(f"\n  Full results: {traj_path}")
    print(f"  Summary:      {summary_path}")
    print("========================================\n")


def parse():
    EVAL_DIR = str(Path(__file__).parent)
    p = argparse.ArgumentParser(description="VideoDR evaluation")
    p.add_argument("--csv",            default=None,
                   help="CSV benchmark (columns: video_id, question, answer, category, difficulty)")
    p.add_argument("--jsonl",          default=None,
                   help="JSONL benchmark (keys: id|video_id, question, label|answer, "
                        "optional category/difficulty/images). Takes precedence over --csv.")
    p.add_argument("--format",         choices=["auto", "jsonl", "csv"], default="auto",
                   help="Override input-format detection. 'auto' (default) prefers --jsonl, "
                        "then infers from --csv extension; defaults to csv if extension is unknown.")
    p.add_argument("--frames-dir",     default=f"{EVAL_DIR}/output/frames",
                   help="Root dir of extracted frames ({frames_dir}/{video_id}/frame_*.png). "
                        "Only consulted when a row's `images` field is empty/absent.")
    p.add_argument("--config",         default=f"{EVAL_DIR}/config.yaml")
    p.add_argument("--system-prompt-file", help="Path to eval_system_prompt.txt; defaults to sibling of --config")
    p.add_argument("--hf-checkpoint",  required=True)
    p.add_argument("--sglang-url",     default="http://localhost:13141")
    p.add_argument("--reward-model-url")
    p.add_argument("--output-dir",     default=f"{EVAL_DIR}/output/results")
    p.add_argument("--max-async-samples", type=int, default=4)
    p.add_argument("--sample-timeout", type=float)
    p.add_argument("--limit",          type=int)
    p.add_argument("--start-index",    type=int, default=0)
    p.add_argument("--video-ids",      help="Comma-separated subset of video IDs to evaluate")
    p.add_argument("--max-turns",      type=int)
    p.add_argument("--rollout-max-context-len", type=int)
    p.add_argument("--max-new-tokens", type=int)
    p.add_argument("--temperature",    type=float)
    p.add_argument("--top-p",          type=float)
    p.add_argument("--top-k",          type=int)
    p.add_argument("--reward-timeout", type=float)
    p.add_argument("--reward-retry-attempts", type=int)
    p.add_argument("--reward-retry-delay",    type=float)
    p.add_argument("--reward-temperature",    type=float)
    p.add_argument("--reward-max-new-tokens", type=int)
    p.add_argument("--model-name", help="Name tag for output subdir (defaults to basename of --hf-checkpoint)")
    p.add_argument("--backend", choices=["sglang", "openai", "vllm", "claude", "gemini", "gpt"], default="sglang",
                   help="Rollout backend: 'sglang' uses /generate API; 'openai'/'vllm' use chat completions; "
                        "'claude' uses a Bedrock-style invoke endpoint; "
                        "'gemini' uses a Google generateContent endpoint; "
                        "'gpt' uses an Azure-style chat-completions endpoint with api-key header")
    p.add_argument("--skip-reward", action="store_true",
                   help="Skip the reward/judge call entirely — only run trajectories. "
                        "is_correct will be left as None and no reward backend is initialized.")
    p.add_argument("--reward-backend", choices=["sglang", "openai", "vllm"], default=None,
                   help="Reward/judge backend (defaults to --backend)")
    p.add_argument("--reward-model",   default=None,
                   help="Model id for the reward endpoint when --reward-backend is "
                        "openai/vllm. Falls back to --vllm-model / --openai-model if unset.")
    p.add_argument("--reward-api-key", default=None,
                   help="Bearer token for the reward endpoint when --reward-backend is "
                        "openai/vllm. Falls back to --openai-api-key if unset.")
    p.add_argument("--openai-api-key",  help="OpenAI-compatible API key")
    p.add_argument("--openai-base-url", help="OpenAI-compatible base URL (e.g. https://<maas-host>/v1)")
    p.add_argument("--openai-model",    help="Model name for OpenAI backend")
    p.add_argument("--vllm-url",        help="vLLM server base URL (e.g. http://localhost:8000/v1)")
    p.add_argument("--vllm-model",      help="Model name as registered in vLLM (e.g. /path/to/model or alias)")
    # Claude / Bedrock-style backend
    p.add_argument("--claude-url",
                   help="Claude Bedrock invoke URL (e.g. https://runway.devops.rednote.life/openai/bedrock_runtime/model/invoke)")
    p.add_argument("--claude-token",
                   help="`token` header value for the Claude endpoint (falls back to env CLAUDE_TOKEN)")
    p.add_argument("--claude-anthropic-version", default=None,
                   help="anthropic_version field (default bedrock-2023-05-31)")
    p.add_argument("--claude-model", default=None,
                   help="Optional `model` field placed in the request body "
                        "(some proxies require this; raw Bedrock invoke does not)")
    p.add_argument("--claude-thinking", action="store_true",
                   help="Enable extended thinking on the Claude endpoint")
    p.add_argument("--claude-thinking-budget", type=int, default=None,
                   help="thinking.budget_tokens (default 4000, must be < max_tokens)")
    p.add_argument("--claude-max-image-dim", type=int, default=None,
                   help=f"Max image side length; larger sides are resized down (default {_DEFAULT_CLAUDE_MAX_IMAGE_DIM})")
    p.add_argument("--claude-max-image-bytes", type=int, default=None,
                   help=f"Max bytes per image after JPEG re-encode (default {_DEFAULT_CLAUDE_MAX_IMAGE_BYTES} ~3.6MB)")
    p.add_argument("--claude-timeout", type=float, default=None,
                   help="HTTP timeout in seconds for each Claude call (default 600)")
    # Gemini / Google generateContent backend
    p.add_argument("--gemini-url",
                   help="Gemini generateContent endpoint "
                        "(e.g. https://runway.devops.rednote.life/openai/google/v1:generateContent)")
    p.add_argument("--gemini-api-key",
                   help="`api-key` header value (falls back to env GEMINI_API_KEY)")
    p.add_argument("--gemini-model", default=None,
                   help="Optional `model` field placed in the request body")
    p.add_argument("--gemini-thinking-budget", type=int, default=None,
                   help="thinkingConfig.thinkingBudget (e.g. 128 for gemini-2.5-pro)")
    p.add_argument("--gemini-thinking-level", default=None,
                   help="thinkingConfig.thinkingLevel — HIGH/MEDIUM/LOW (gemini-3 only). "
                        "If set, overrides --gemini-thinking-budget.")
    p.add_argument("--gemini-include-thoughts", action="store_true",
                   help="thinkingConfig.includeThoughts — include reasoning trace in response")
    p.add_argument("--gemini-max-image-dim", type=int, default=None,
                   help=f"Max image side length (default {_DEFAULT_GEMINI_MAX_IMAGE_DIM})")
    p.add_argument("--gemini-max-image-bytes", type=int, default=None,
                   help=f"Max bytes per image after JPEG re-encode (default {_DEFAULT_GEMINI_MAX_IMAGE_BYTES} ~3.6MB)")
    p.add_argument("--gemini-timeout", type=float, default=None,
                   help="HTTP timeout in seconds for each Gemini call (default 600)")
    # GPT / Azure-style chat completions backend
    p.add_argument("--gpt-url",
                   help="Chat completions endpoint, e.g. "
                        "https://runway.devops.rednote.life/openai/chat/completions?api-version=2024-12-01-preview")
    p.add_argument("--gpt-api-key",
                   help="`api-key` header value (falls back to env GPT_API_KEY)")
    p.add_argument("--gpt-model", default=None,
                   help="Optional `model` field placed in the request body "
                        "(some Azure endpoints accept the deployment name, others don't need it)")
    p.add_argument("--gpt-max-image-dim", type=int, default=None,
                   help=f"Max image side length (default {_DEFAULT_GPT_MAX_IMAGE_DIM})")
    p.add_argument("--gpt-max-image-bytes", type=int, default=None,
                   help=f"Max bytes per image after JPEG re-encode (default {_DEFAULT_GPT_MAX_IMAGE_BYTES} ~3.6MB)")
    p.add_argument("--gpt-timeout", type=float, default=None,
                   help="HTTP timeout in seconds for each GPT call (default 600)")
    p.add_argument("--mode", choices=["tool", "direct"], default="tool",
                   help="'tool': multi-turn deep research (no crop_and_search); "
                        "'direct': single-turn answer from keyframes only")
    p.add_argument("--log-level",      default="INFO")
    return p.parse_args()


if __name__ == "__main__":
    cli = parse()
    logging.basicConfig(
        level=getattr(logging, cli.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    if cli.max_async_samples <= 0:
        raise ValueError("--max-async-samples must be positive")
    asyncio.run(main_async(cli))
