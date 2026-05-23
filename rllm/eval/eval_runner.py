"""
DeepResearch evaluation runner (standalone).

Mirrors the execution semantics of the reference run_eval11.sh / eval_runner.py:
- Same CLI surface (--parquet --base-url --model --parallel-tasks --api-key ...).
- Same output layout: <output_dir>/<timestamp>/{episodes.jsonl, metrics.json, config.json}.
- Same episode JSON shape (id/task/termination_reason/is_correct/metrics/info/trajectories[]).
- Same judge prompt (canonical "deep research report contains the correct answer" form).

Differences from the reference (these are the only ones, dictated by env constraints):
- No `rllm` dependency — agent loop runs via `AsyncOpenAI` directly.
- Parquet is loaded with pandas (handles numpy arrays in the `images` column).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Optional

import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from PIL import Image

# Tools live next to this file (self-contained — no rllm / no parent-dir hop).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tools import get_all_tools

from deepresearch_agent import (
    MultiTurnReactAgent,
    DEEPRESEARCH_SYSTEM_PROMPT,
    DEEPRESEARCH_SYSTEM_PROMPT_TEXT,
)


# ---------------------- Image utils ---------------------- #


def _read_bytes(img: Any) -> Optional[bytes]:
    if isinstance(img, str):
        return open(img, "rb").read() if os.path.isfile(img) else None
    if isinstance(img, bytes):
        return img
    if isinstance(img, dict):
        if img.get("bytes"):
            return img["bytes"]
        path = img.get("path")
        if path and os.path.isfile(path):
            return open(path, "rb").read()
    return None


def load_image_as_base64_url(img: Any) -> Optional[str]:
    """Convert an image entry (path / bytes / dict / np scalar) to a base64 data URI."""
    try:
        if isinstance(img, np.ndarray):
            img = img.item() if img.size == 1 else img.tolist()
        data = _read_bytes(img)
        if not data:
            return None
        pil_img = Image.open(io.BytesIO(data))
        fmt = (pil_img.format or "JPEG").upper()
        mime = f"image/{fmt.lower()}"
        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except Exception as e:
        print(f"Warning: failed to load image: {e}")
        return None


def _get_image_path(img: Any) -> Optional[str]:
    """Extract a local filesystem path (needed by `crop_and_search`)."""
    if isinstance(img, np.ndarray):
        img = img.item() if img.size == 1 else (img.tolist()[0] if img.size else None)
    if isinstance(img, str) and os.path.isfile(img):
        return img
    if isinstance(img, dict):
        path = img.get("path")
        if path and os.path.isfile(path):
            return path
    return None


# ---------------------- Data loading ---------------------- #


def _coerce_images(raw: Any) -> List[Any]:
    """Normalize the `images` cell from a pandas row into a list."""
    if raw is None:
        return []
    if isinstance(raw, float) and pd.isna(raw):
        return []
    if isinstance(raw, np.ndarray):
        return [x for x in raw.tolist() if x is not None]
    if isinstance(raw, list):
        return [x for x in raw if x is not None]
    return [raw]


def load_tasks(args) -> List[dict]:
    if not args.parquet:
        raise ValueError("Parquet path is required. Please set --parquet.")

    df = pd.read_parquet(str(args.parquet))

    required = {"question", "answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Parquet file missing required columns: {sorted(missing)}")

    if args.max_samples is not None:
        df = df.head(args.max_samples)

    tasks: List[dict] = []
    for idx, row in df.iterrows():
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        if not question:
            raise ValueError(f"Record {idx} has empty question")
        if not answer:
            raise ValueError(f"Record {idx} has empty answer")

        images = _coerce_images(row.get("images", None) if "images" in df.columns else None)

        tid = None
        for key in ("id", "idx", "_id"):
            if key in df.columns and row.get(key) is not None:
                tid = row.get(key)
                break
        if tid is None:
            tid = int(idx) if isinstance(idx, (int, np.integer)) else str(idx)

        tasks.append({
            "id": tid,
            "question": question,
            "answer": answer,
            "images": images,
        })

    if not tasks:
        raise ValueError("No valid tasks loaded from Parquet.")
    return tasks


# ---------------------- Judge (verbatim canonical prompt) ---------------------- #


JUDGE_PROMPT = """You are a impartial judge evaluating whether a deep research report contains the correct answer.

[Question]
{question}

[Correct Answer]
{reference_answer}

[Deep Research Report]
{assistant_answer}

Task: Determine if the deep research report contains the correct answer anywhere in its content.

Instructions:
1. Read through the entire research report carefully
2. Look for the correct answer anywhere in the report (it may be embedded in paragraphs, tables, or sections)
3. Check if the information in the report is consistent with the correct answer
4. The answer does NOT need to be in a specific format or labeled as "final answer"
5. Provide your reasoning
6. Answer with "yes" if the report contains the correct answer, "no" if it doesn't or contradicts it

Output format:
correct: [yes/no]
reasoning: [your explanation]
"""


def _parse_judge_response(text: str) -> Optional[bool]:
    if not text:
        return None
    for line in text.lower().splitlines():
        line = line.strip()
        if "correct:" in line:
            after = line.split("correct:", 1)[1].strip()
            if after.startswith("yes"):
                return True
            if after.startswith("no"):
                return False
            if "yes" in after:
                return True
            if "no" in after:
                return False
    return None


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower()).strip(".,;:!?\"'")


def _extract_final_answer(response: str) -> str:
    """Strip <think>, then prefer <answer>, fall back to \\boxed{}, else full text."""
    if not response:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    m = re.search(r"<answer>(.*?)</answer>", cleaned, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"\\boxed\{([^{}]+)\}", cleaned)
    if m:
        return m.group(1).strip()
    return cleaned.strip()


async def judge_answer(
    client: AsyncOpenAI,
    judge_model: str,
    question: str,
    ground_truth: str,
    prediction: str,
) -> tuple[bool, str]:
    """Returns (is_correct, raw judge text). Falls back to normalized exact match."""
    try:
        prompt = JUDGE_PROMPT.format(
            question=question,
            reference_answer=ground_truth,
            assistant_answer=prediction,
        )
        resp = await client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        text = (resp.choices[0].message.content or "").strip()
        verdict = _parse_judge_response(text)
        if verdict is None:
            return _normalize(ground_truth) == _normalize(prediction), text
        return verdict, text
    except Exception as e:
        print(f"Judge LLM failed: {e}, falling back to normalized exact match")
        return _normalize(ground_truth) == _normalize(prediction), f"judge_error: {e}"


# ---------------------- Task execution ---------------------- #


async def run_task(
    task: dict,
    client: AsyncOpenAI,
    judge_client: AsyncOpenAI,
    judge_model: str,
    tools: dict,
    args,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Run a single task through the agent + judge; return reference-shaped episode dict."""
    async with semaphore:
        task_id = task.get("id", "?")
        question = task["question"]
        answer = task["answer"]
        images_raw = task.get("images", [])

        image_urls: List[str] = []
        image_path: Optional[str] = None
        for img in images_raw:
            url = load_image_as_base64_url(img)
            if url:
                image_urls.append(url)
                if image_path is None:
                    image_path = _get_image_path(img)

        system_prompt = (
            DEEPRESEARCH_SYSTEM_PROMPT if image_urls else DEEPRESEARCH_SYSTEM_PROMPT_TEXT
        )

        sampling_params: dict = {
            "temperature": args.temperature,
            "top_p": args.top_p,
        }
        if args.max_tokens is not None:
            sampling_params["max_tokens"] = args.max_tokens

        agent = MultiTurnReactAgent(
            client=client,
            model=args.model,
            sampling_params=sampling_params,
            tools=tools,
            system_prompt=system_prompt,
        )

        print(f"[Task {task_id}] Starting...")
        agent_result = await agent.run(
            question=question,
            answer=answer,
            images=image_urls or None,
            image_path=image_path,
        )

        raw_prediction = agent_result.get("prediction", "") or ""
        prediction = _extract_final_answer(raw_prediction)

        is_correct, judge_text = await judge_answer(
            judge_client, judge_model, question, answer, prediction
        )

        print(
            f"[Task {task_id}] Done. correct={is_correct}, "
            f"termination={agent_result.get('termination')}, "
            f"rounds={agent_result.get('rounds')}"
        )

        return _to_episode_dict(
            task=task,
            agent_result=agent_result,
            prediction=prediction,
            is_correct=is_correct,
            judge_text=judge_text,
            image_urls=image_urls,
        )


# ---------------------- Serialization (match reference shape) ---------------------- #


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, Image.Image):
        return {"type": "PIL.Image", "size": obj.size}
    if isinstance(obj, bytes):
        return f"<bytes:{len(obj)}>"
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def _strip_base64(messages: List[dict]) -> List[dict]:
    """Replace base64 image data URIs with a short placeholder to keep jsonl readable."""
    out = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            new_content = []
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "image_url"
                    and isinstance(part.get("image_url"), dict)
                ):
                    url = part["image_url"].get("url", "")
                    if isinstance(url, str) and url.startswith("data:"):
                        new_content.append({"type": "image_url", "image_url": {"url": "<base64-image>"}})
                        continue
                new_content.append(part)
            new_msg = dict(msg)
            new_msg["content"] = new_content
            out.append(new_msg)
        else:
            out.append(msg)
    return out


def _to_episode_dict(
    *,
    task: dict,
    agent_result: dict,
    prediction: str,
    is_correct: bool,
    judge_text: str,
    image_urls: List[str],
) -> dict:
    """Render the result in the reference episode shape (single-trajectory, single-step)."""
    messages = _strip_base64(agent_result.get("messages", []) or [])
    termination = agent_result.get("termination") or "unknown"
    reward = 1.0 if is_correct else 0.0

    task_view = {
        "id": task.get("id"),
        "question": task.get("question"),
        "answer": task.get("answer"),
        "num_images": len(image_urls),
    }

    step = {
        "chat_completions": _sanitize(messages),
        "model_response": _sanitize(agent_result.get("prediction", "")),
        "action": _sanitize(prediction),
        "observation": None,
        "reward": reward,
    }

    trajectory = {
        "name": "deepresearch",
        "reward": reward,
        "info": {
            "judge_response": judge_text,
            "rounds": agent_result.get("rounds", 0),
            "time_taken": agent_result.get("time_taken", 0),
            "token_usage": _sanitize(agent_result.get("token_usage", {})),
        },
        "steps": [step],
    }

    return {
        "id": task.get("id"),
        "task": _sanitize(task_view),
        "termination_reason": termination,
        "is_correct": bool(is_correct),
        "metrics": {
            "reward": reward,
            "rounds": agent_result.get("rounds", 0),
        },
        "info": {
            "prediction": prediction,
            "judge_response": judge_text,
        },
        "trajectories": [trajectory],
    }


# ---------------------- Metrics & IO ---------------------- #


def compute_metrics(episodes: Iterable[dict]) -> dict:
    episodes = list(episodes)
    total = len(episodes)
    correct = sum(1 for ep in episodes if ep.get("is_correct"))
    termination: dict = {}
    rewards: List[float] = []
    for ep in episodes:
        reason = ep.get("termination_reason") or "unknown"
        termination[reason] = termination.get(reason, 0) + 1
        for tr in ep.get("trajectories", []) or []:
            r = tr.get("reward")
            if r is not None:
                rewards.append(float(r))
    avg_reward = (sum(rewards) / len(rewards)) if rewards else None
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "termination_distribution": termination,
        "average_reward": avg_reward,
    }


def save_outputs(
    episodes: List[dict], metrics: dict, output_dir: Path, config: dict
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ep_path = output_dir / "episodes.jsonl"
    with ep_path.open("w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Saved episodes to {ep_path}")
    print(f"Metrics: {metrics}")


# ---------------------- Arg parsing ---------------------- #


def parse_args():
    parser = argparse.ArgumentParser(
        description="DeepResearch evaluation (standalone, OpenAI-compatible)"
    )

    # Data
    parser.add_argument("--parquet", default=None, help="Local Parquet path (required)")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples")

    # Rollout
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max completion tokens")

    # Judge
    parser.add_argument("--judge-model", default=None, help="Judge model (default: same as --model)")
    parser.add_argument("--judge-base-url", default=None, help="Judge base URL (default: same as --base-url)")
    parser.add_argument("--judge-api-key", default=None, help="Judge API key (default: same as --api-key)")

    # Execution
    parser.add_argument("--parallel-tasks", type=int, default=10, help="Parallel tasks")
    parser.add_argument("--retry-limit", type=int, default=1, help="(accepted for ref-compat; no-op)")
    parser.add_argument("--output-dir", default="./outputs", help="Output directory")

    return parser.parse_args()


# ---------------------- Main ---------------------- #


async def main():
    args = parse_args()

    # Fill from environment when CLI args are absent. No hardcoded keys.
    args.model = args.model or os.getenv("TONGYI_MODEL_NAME")
    args.base_url = args.base_url or os.getenv("TONGYI_BASE_URL")
    args.api_key = args.api_key or os.getenv("TONGYI_API_KEY") or "EMPTY"

    if not args.model:
        raise ValueError("--model (or TONGYI_MODEL_NAME) is required")
    if not args.base_url:
        raise ValueError("--base-url (or TONGYI_BASE_URL) is required")

    judge_model = args.judge_model or os.getenv("JUDGE_MODEL") or args.model
    judge_base_url = args.judge_base_url or os.getenv("JUDGE_BASE_URL") or args.base_url
    judge_api_key = args.judge_api_key or os.getenv("JUDGE_API_KEY") or args.api_key

    tasks = load_tasks(args)
    print(f"Running DeepResearch evaluation with {len(tasks)} tasks")

    tools = get_all_tools()

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    judge_client = AsyncOpenAI(base_url=judge_base_url, api_key=judge_api_key)

    semaphore = asyncio.Semaphore(args.parallel_tasks)

    coroutines = [
        run_task(task, client, judge_client, judge_model, tools, args, semaphore)
        for task in tasks
    ]
    raw_results = await asyncio.gather(*coroutines, return_exceptions=True)

    episodes: List[dict] = []
    for i, r in enumerate(raw_results):
        if isinstance(r, Exception):
            print(f"Task {i} failed with exception: {r}")
            episodes.append({
                "id": tasks[i].get("id", i),
                "task": _sanitize({
                    "id": tasks[i].get("id"),
                    "question": tasks[i].get("question"),
                    "answer": tasks[i].get("answer"),
                }),
                "termination_reason": f"error: {r}",
                "is_correct": False,
                "metrics": {"reward": 0.0, "rounds": 0},
                "info": {"prediction": "", "judge_response": ""},
                "trajectories": [],
            })
        else:
            episodes.append(r)

    metrics = compute_metrics(episodes)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp

    config = {
        "model": args.model,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "parallel_tasks": args.parallel_tasks,
        "parquet": str(args.parquet) if args.parquet else None,
        "max_samples": args.max_samples,
        "judge_model": judge_model,
        "judge_base_url": judge_base_url,
    }

    save_outputs(episodes, metrics, output_dir, config)


if __name__ == "__main__":
    asyncio.run(main())
