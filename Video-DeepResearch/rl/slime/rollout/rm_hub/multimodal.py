import os

import aiohttp

from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample


async def call_multimodal_teacher(
    args,
    sample: Sample,
    *,
    return_text_in_logprobs: bool = False,
    use_teacher_context_env: bool = False,
) -> dict:
    """Call SGLang teacher server for multimodal token logprobs."""
    if use_teacher_context_env and os.environ.get("USE_TEACHER_CONTEXT", "0") == "1":
        if "teacher_prompt_ids" in sample.metadata:
            response_ids = sample.tokens[-sample.response_length:]
            input_ids = list(sample.metadata["teacher_prompt_ids"]) + list(response_ids)
            teacher_images = sample.metadata.get(
                "teacher_images",
                (sample.multimodal_inputs or {}).get("images", []),
            )
        else:
            input_ids = sample.tokens
            teacher_images = (sample.multimodal_inputs or {}).get("images", [])
    else:
        input_ids = sample.tokens
        teacher_images = (sample.multimodal_inputs or {}).get("images", [])

    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    if return_text_in_logprobs:
        payload["return_text_in_logprobs"] = True
    if teacher_images:
        payload["image_data"] = [encode_image_for_rollout_engine(img) for img in teacher_images]

    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


def _extract_boxed_answer(response: str) -> str | None:
    """Extract \\boxed{...} answer from response if present."""
    import re
    match = re.search(r'\\boxed\{([^}]+)\}', response)
    if match:
        return match.group(1)
    return None


_JUDGE_MODEL_CACHE: dict[str, str] = {}


def _judge_base_url(judge_url: str) -> str:
    """Normalize `args.judge_url` to a base origin (no path)."""
    base = judge_url.rstrip("/")
    for suffix in ("/v1/chat/completions", "/chat/completions", "/generate", "/v1"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base


async def _resolve_judge_model(base_url: str) -> str:
    if base_url in _JUDGE_MODEL_CACHE:
        return _JUDGE_MODEL_CACHE[base_url]
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(f"{base_url}/v1/models") as resp:
            resp.raise_for_status()
            data = await resp.json()
    model_id = data["data"][0]["id"]
    _JUDGE_MODEL_CACHE[base_url] = model_id
    return model_id


async def call_llm_judge(args, sample: Sample) -> float:
    """Score correctness with a fast-path math rule judge, falling back to a
    vLLM judge over the OpenAI-compatible /v1/chat/completions API.

    Strategy:
    1. Try fast math rule judge first (compute_math_reward)
    2. If correct, return 1.0 immediately (no LLM call)
    3. If incorrect, fall back to vLLM judge for verdict
    4. LLM judge extracts \\boxed{...} if present, otherwise uses full response
    """
    try:
        math_reward = compute_math_reward(sample)
        if math_reward > 0.5:
            return 1.0
    except Exception:
        pass

    question = sample.metadata.get("prompt", str(sample.label))
    ground_truth = str(sample.label)
    prediction = sample.response

    boxed_answer = _extract_boxed_answer(prediction)
    if boxed_answer:
        prediction_for_judge = f"\\boxed{{{boxed_answer}}}"
    else:
        prediction_for_judge = prediction

    judge_prompt = (
        "Please evaluate whether the model's answer is correct by comparing it "
        "with the standard answer.\n"
        f"Question: {question}\n"
        f"Ground Truth Answer: {ground_truth}\n"
        f"Predicted Answer: {prediction_for_judge}\n\n"
        "**Instructions:**\n"
        "- Compare the model's answer with the standard answer\n"
        "- Focus on factual accuracy and key points\n"
        "- Allow for different wording if the core meaning is the same\n"
        "- Consider the answer correct if it captures the main points\n"
        "- Output only one word: \"correct\" or \"incorrect\".\n\n"
        "**Output format:**\ncorrect/incorrect"
    )

    base = _judge_base_url(args.judge_url)
    model = await _resolve_judge_model(base)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": judge_prompt}],
        "max_tokens": 8192,
        "temperature": 0,
    }

    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(f"{base}/v1/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            result = await resp.json()

    content = (result["choices"][0]["message"]["content"] or "").strip().lower()
    if "incorrect" in content:
        return 0.0
    if "correct" in content:
        return 1.0
    return 0.0


def compute_math_reward(sample: Sample) -> float:
    """Rule-based math reward for a sample."""
    from slime.rollout.rm_hub.math_utils import grade_answer_verl

    return 1.0 if grade_answer_verl(sample.response, str(sample.label)) else 0.0

