#!/usr/bin/env python3
"""
VideoDR evaluation via OpenAI-compatible API (XiaoHongshu MaaS / any OpenAI endpoint).

No local HF checkpoint needed — all inference goes through the remote API.

Usage:
  python3 run_eval_maas.py \
    --csv     /path/to/VideoDR.csv \
    --frames-dir /path/to/frames \
    --api-key   <token> \
    --base-url  https://<maas-host>/v1 \
    --model     qwen3.5-35b-a3b \
    --mode      direct \
    --output-dir /path/to/results \
    --model-name my-model-tag
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import datetime
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

log = logging.getLogger(__name__)

ANSWER_RE   = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


# ---------------------------------------------------------------------------
# OpenAI client helpers
# ---------------------------------------------------------------------------

def _make_client(api_key: str, base_url: str, headers: dict):
    from openai import AsyncOpenAI
    return AsyncOpenAI(api_key=api_key, base_url=base_url, default_headers=headers)


def _img_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _build_openai_messages(image_paths: list[str], question: str,
                            system_prompt: str, include_images: bool) -> list[dict]:
    user_parts: list = []
    if include_images and image_paths:
        user_parts.append({"type": "text",
                            "text": "The following are keyframes sampled from a video clip:\n"})
        for p in image_paths:
            if os.path.exists(p):
                b64 = _img_to_b64(p)
                user_parts.append({"type": "image_url",
                                   "image_url": {"url": f"data:image/png;base64,{b64}"}})
        user_parts.append({"type": "text", "text": f"\n\nQuestion: {question}"})
    else:
        user_parts = [{"type": "text", "text": f"Question: {question}"}]

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_parts},
    ]


# ---------------------------------------------------------------------------
# CSV / frame helpers (same as run_eval.py)
# ---------------------------------------------------------------------------

def load_csv(path: str) -> list[dict]:
    records = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or not row[0].strip():
                continue
            records.append({
                "video_id":   row[0].strip(),
                "question":   row[1].strip(),
                "answer":     row[2].strip(),
                "category":   row[3].strip() if len(row) > 3 else "",
                "difficulty": row[4].strip() if len(row) > 4 else "",
            })
    return records


def find_frames(frames_dir: str, video_id: str) -> list[str]:
    d = Path(frames_dir) / video_id
    return sorted(str(p) for p in d.glob("frame_*.png")) if d.exists() else []


def load_existing(path: str) -> set[int]:
    out = Path(path)
    if not out.exists():
        return set()
    done: set[int] = set()
    for line in out.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            if isinstance(r.get("index"), int):
                done.add(r["index"])
        except json.JSONDecodeError:
            pass
    return done


# ---------------------------------------------------------------------------
# System prompt building
# ---------------------------------------------------------------------------

def _build_system_direct() -> str:
    return (
        "Answer the question based on the keyframes provided. "
        "Place your final answer within <answer></answer> tags."
    )


def _build_system_tool(prompt_file: str, supported_tools: list[str]) -> str:
    from vdr_core.tools.registry import get_tools
    tool_map = {t["function"]["name"]: t for t in get_tools()}
    tools_json = "\n".join(
        json.dumps(tool_map[n], ensure_ascii=False)
        for n in supported_tools if n in tool_map
    )
    date_str = datetime.date.today().isoformat()
    template = Path(prompt_file).read_text(encoding="utf-8")
    return template.replace("{tools}", tools_json).replace("{date}", date_str)


# ---------------------------------------------------------------------------
# Reward (reuse existing DeepResearchReward)
# ---------------------------------------------------------------------------

def _make_reward(c):
    from vdr_core.deepresearch_reward import DeepResearchReward
    reward_url = getattr(c, "reward_url", None)
    reward_backend = getattr(c, "reward_backend", None) or ("sglang" if reward_url else "openai")
    return DeepResearchReward(
        backend=reward_backend,
        url=reward_url,
        openai_api_key=c.api_key,
        openai_base_url=c.base_url,
        openai_model=c.reward_model or c.model,
        openai_headers=_parse_headers(c.header),
        max_new_tokens=512,
        temperature=0.0,
    )


def _parse_headers(pairs: list[str] | None) -> dict:
    h: dict = {}
    for p in (pairs or []):
        k, _, v = p.partition("=")
        h[k.strip()] = v.strip()
    return h


# ---------------------------------------------------------------------------
# Single-sample async evaluation
# ---------------------------------------------------------------------------

async def run_one(idx: int, rec: dict, frames: list[str],
                  client, model: str, system_prompt: str,
                  include_images: bool, max_tokens: int, temperature: float,
                  reward_fn, sem: asyncio.Semaphore, timeout: float | None,
                  mode: str) -> dict:
    async with sem:
        response_text, e_str = "", None
        try:
            messages = _build_openai_messages(
                frames, rec["question"], system_prompt, include_images
            )
            coro = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
            )
            resp = await (asyncio.wait_for(coro, timeout=timeout) if timeout else coro)
            response_text = resp.choices[0].message.content or ""
        except Exception as exc:
            import traceback
            e_str = traceback.format_exc()
            log.error("sample %d failed: %s", idx, exc)

        rollout_ok = not bool(e_str)

        # Score
        is_correct, reward_call_ok = None, True
        try:
            result = await reward_fn.async_score(
                question=rec["question"],
                response=response_text,
                answer=rec["answer"],
            )
            is_correct = result.get("is_correct")
        except Exception:
            reward_call_ok = False

        log.info("video=%-6s  is_correct=%s  rollout_ok=%s",
                 rec["video_id"], is_correct, rollout_ok)
        return {
            "index":          idx,
            "video_id":       rec["video_id"],
            "category":       rec["category"],
            "difficulty":     rec["difficulty"],
            "rollout_ok":     rollout_ok,
            "reward_call_ok": reward_call_ok,
            "is_correct":     is_correct,
            "answer":         response_text,
            "label":          rec["answer"],
            "exception":      e_str,
        }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def compute_summary(results: list[dict]) -> dict:
    by_cat: dict[str, list] = defaultdict(list)
    by_diff: dict[str, list] = defaultdict(list)
    scored = 0
    for r in results:
        ic = r.get("is_correct")
        if ic is None:
            continue
        scored += 1
        v = 1 if ic else 0
        by_cat[r.get("category", "?")].append(v)
        by_diff[r.get("difficulty", "?")].append(v)

    def acc(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    all_v = [v for lst in by_cat.values() for v in lst]
    return {
        "total":         len(results),
        "scored":        scored,
        "accuracy":      acc(all_v),
        "by_category":   {k: {"n": len(v), "accuracy": acc(v)} for k, v in sorted(by_cat.items())},
        "by_difficulty": {k: {"n": len(v), "accuracy": acc(v)} for k, v in sorted(by_diff.items())},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(c):
    headers = _parse_headers(c.header)
    client  = _make_client(c.api_key, c.base_url, headers)
    reward_fn = _make_reward(c)

    # System prompt
    if c.mode == "direct":
        system_prompt  = _build_system_direct()
        include_images = True
    else:
        eval_dir      = Path(__file__).parent
        prompt_file   = c.system_prompt_file or str(eval_dir / "eval_system_prompt.txt")
        supported     = [t.strip() for t in c.supported_tools.split(",")]
        system_prompt = _build_system_tool(prompt_file, supported)
        include_images = True

    # Load data
    if not c.csv:
        raise ValueError("--csv is required (path to VideoDR.csv)")
    if not c.frames_dir:
        raise ValueError("--frames-dir is required (path to the extracted-keyframes root)")
    records = load_csv(c.csv)
    if c.video_ids:
        keep = set(c.video_ids.split(","))
        records = [r for r in records if r["video_id"] in keep]
    log.info("Loaded %d records", len(records))

    # Output
    model_tag  = c.model_name or c.model.replace("/", "_")
    output_dir = Path(c.output_dir) / model_tag / c.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    traj_path    = output_dir / "trajectories.jsonl"
    summary_path = output_dir / "summary.json"

    existing = load_existing(str(traj_path))
    entries = [
        (i, rec, find_frames(c.frames_dir, rec["video_id"]))
        for i, rec in enumerate(records)
        if i not in existing
    ]
    skipped = len(records) - len(entries)
    if skipped:
        log.info("Skipping %d already-done samples", skipped)

    sem    = asyncio.Semaphore(c.max_async_samples)
    tasks  = [
        asyncio.create_task(run_one(
            idx, rec, frames, client, c.model,
            system_prompt, include_images,
            c.max_new_tokens, c.temperature,
            reward_fn, sem, c.timeout, c.mode,
        ))
        for idx, rec, frames in entries
    ]

    all_results: list[dict] = []
    with open(str(traj_path), "a", encoding="utf-8") as f:
        for fut in asyncio.as_completed(tasks):
            row = await fut
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            all_results.append(row)

    summary = compute_summary(all_results)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n========== Evaluation Summary ==========")
    print(f"  Model:    {model_tag}  mode={c.mode}")
    print(f"  Total:    {summary['total']}")
    print(f"  Scored:   {summary['scored']}")
    print(f"  Accuracy: {summary['accuracy']}")
    print("\n  By category:")
    for k, v in summary["by_category"].items():
        print(f"    {k:<20} n={v['n']}  acc={v['accuracy']}")
    print("\n  By difficulty:")
    for k, v in summary["by_difficulty"].items():
        print(f"    {k:<10} n={v['n']}  acc={v['accuracy']}")
    print(f"\n  Results: {traj_path}")
    print("========================================\n")


def parse_args():
    EVAL_DIR = str(Path(__file__).parent)
    p = argparse.ArgumentParser(description="VideoDR eval via OpenAI-compatible API")
    p.add_argument("--csv",        default=None)
    p.add_argument("--frames-dir", default=None)
    p.add_argument("--output-dir", default=f"{EVAL_DIR}/output/results")
    p.add_argument("--api-key",    required=True)
    p.add_argument("--base-url",   required=True)
    p.add_argument("--model",      required=True, help="Model name for inference (e.g. qwen3.5-35b-a3b)")
    p.add_argument("--reward-url",   help="SGLang /generate URL for judge (e.g. http://10.x.x.x:13141); if set, --reward-backend defaults to sglang")
    p.add_argument("--reward-model", help="Model name for judging when using openai reward backend (defaults to --model)")
    p.add_argument("--reward-backend", choices=["sglang", "openai"], default=None,
                   help="Judge backend: 'sglang' uses --reward-url, 'openai' uses --base-url (default: sglang if --reward-url set, else openai)")
    p.add_argument("--header",     action="append", metavar="KEY=VALUE",
                   help="Extra HTTP headers, e.g. x-maas-user-email=foo@bar.com  (repeatable)")
    p.add_argument("--mode",       choices=["direct", "tool"], default="direct")
    p.add_argument("--model-name", help="Tag for output subdir (defaults to --model)")
    p.add_argument("--supported-tools", default="search,visit,select_crop_search")
    p.add_argument("--system-prompt-file")
    p.add_argument("--max-async-samples", type=int,   default=4)
    p.add_argument("--max-new-tokens",    type=int,   default=4096)
    p.add_argument("--temperature",       type=float, default=0.0)
    p.add_argument("--timeout",           type=float)
    p.add_argument("--video-ids",  help="Comma-separated subset of video IDs")
    p.add_argument("--log-level",  default="INFO")
    return p.parse_args()


if __name__ == "__main__":
    c = parse_args()
    logging.basicConfig(level=getattr(logging, c.log_level),
                        format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(main_async(c))
