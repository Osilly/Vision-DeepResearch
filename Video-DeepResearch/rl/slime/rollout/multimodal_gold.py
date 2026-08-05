"""
Minimal multimodal GOLD-advantage distillation.

This script is intentionally standalone and does not modify existing OPD logic.
It plugs into the same slime hooks as on-policy distillation:

  --custom-rm-path slime.rollout.multimodal_gold.reward_func_math
  --custom-rm-path slime.rollout.multimodal_gold.reward_func_judge
  --custom-rm-path slime.rollout.multimodal_gold.reward_func
  --custom-reward-post-process-path slime.rollout.multimodal_gold.post_process_rewards

Design goals:
- Keep only essential fields
- Teacher essential fields:
  - token log_probs (stored in sample.teacher_log_probs after GOLD projection)
  - decoded token triplets (stored in sample.metadata["teacher_token_triplets"])
- Do basic GOLD flow only:
  - group alignment between student/teacher token pieces
  - save group results for debugging
  - no full-vocab GOLD
- Adapt to slime's existing OPD loss path by projecting group penalty back into
  a synthetic teacher_log_probs tensor on the student token space
- Each student token in a group receives the full group logprob gap
  (not averaged / not evenly split)
- No task reward participates in advantage computation; processed rewards are
  always zeros. Raw rewards can still be computed for supervision/logging.
"""
from __future__ import annotations

import math
import os
import time

import torch
from transformers import AutoTokenizer

from slime.rollout.rm_hub.multimodal import call_llm_judge, compute_math_reward
from slime.utils.processing_utils import build_processor_kwargs, encode_image_for_rollout_engine, load_processor
from slime.utils.teacher_pool import get_teacher_pool
from slime.utils.types import Sample


_TOKENIZER_CACHE: dict[str, AutoTokenizer] = {}
_PROCESSOR_CACHE: dict[str, object | None] = {}


def _get_tokenizer(hf_checkpoint: str):
    tokenizer = _TOKENIZER_CACHE.get(hf_checkpoint)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint, trust_remote_code=True)
        _TOKENIZER_CACHE[hf_checkpoint] = tokenizer
    return tokenizer


def _get_processor(hf_checkpoint: str):
    if hf_checkpoint not in _PROCESSOR_CACHE:
        _PROCESSOR_CACHE[hf_checkpoint] = load_processor(hf_checkpoint, trust_remote_code=True)
    return _PROCESSOR_CACHE[hf_checkpoint]


def _get_student_tokenizer(hf_checkpoint: str):
    return _get_tokenizer(hf_checkpoint)


def _get_teacher_hf_checkpoint(args) -> str:
    teacher_hf_checkpoint = getattr(args, "gold_teacher_hf_checkpoint", None) or os.environ.get("GOLD_TEACHER_HF_CHECKPOINT")
    if not teacher_hf_checkpoint:
        teacher_hf_checkpoint = os.environ.get("TEACHER_HF_CHECKPOINT", "")
    if not teacher_hf_checkpoint:
        raise ValueError(
            "GOLD requires a local teacher tokenizer checkpoint. Set args.gold_teacher_hf_checkpoint "
            "or environment variable GOLD_TEACHER_HF_CHECKPOINT."
        )
    return teacher_hf_checkpoint


def _get_teacher_tokenizer(args):
    return _get_tokenizer(_get_teacher_hf_checkpoint(args))


def _get_teacher_processor(args):
    return _get_processor(_get_teacher_hf_checkpoint(args))


def _build_teacher_prompt_text(args, sample: Sample) -> str:
    raw_prompt = sample.metadata.get("raw_prompt")
    if raw_prompt is None:
        assert isinstance(sample.prompt, str), "GOLD teacher text path expects sample.prompt to be a string"
        return sample.prompt

    tools = sample.metadata.get("tools")
    teacher_tokenizer = _get_teacher_tokenizer(args)
    return teacher_tokenizer.apply_chat_template(
        raw_prompt,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )


def _trim_prompt_eos(tokenizer, prompt_ids: list[int]) -> list[int]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and prompt_ids and prompt_ids[-1] == eos_token_id:
        return prompt_ids[:-1]
    return prompt_ids


def _build_teacher_prompt_ids(args, sample: Sample) -> list[int]:
    prompt_text = _build_teacher_prompt_text(args, sample)
    teacher_processor = _get_teacher_processor(args)
    if teacher_processor is not None and sample.multimodal_inputs and any(
        v is not None for v in sample.multimodal_inputs.values()
    ):
        processor_kwargs = build_processor_kwargs(sample.multimodal_inputs)
        processor_output = teacher_processor(text=prompt_text, **processor_kwargs)
        prompt_ids = processor_output["input_ids"][0]
        if isinstance(prompt_ids, torch.Tensor):
            prompt_ids = prompt_ids.tolist()
        else:
            prompt_ids = list(prompt_ids)
        teacher_tokenizer = getattr(teacher_processor, "tokenizer", None) or _get_teacher_tokenizer(args)
        return _trim_prompt_eos(teacher_tokenizer, prompt_ids)

    teacher_tokenizer = _get_teacher_tokenizer(args)
    prompt_ids = teacher_tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
    return _trim_prompt_eos(teacher_tokenizer, prompt_ids)


def _build_teacher_token_ids(args, sample: Sample) -> tuple[list[int], int]:
    prompt_ids = _build_teacher_prompt_ids(args, sample)
    teacher_tokenizer = _get_teacher_tokenizer(args)
    response_ids = teacher_tokenizer(sample.response, add_special_tokens=False)["input_ids"]
    full_input_ids = list(prompt_ids) + list(response_ids)
    return full_input_ids, len(prompt_ids)


async def _call_teacher(args, sample: Sample) -> dict:
    teacher_input_ids, teacher_prompt_len = _build_teacher_token_ids(args, sample)
    payload = {
        "input_ids": teacher_input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
        "return_text_in_logprobs": True,
    }

    teacher_images = (sample.multimodal_inputs or {}).get("images", [])
    if teacher_images:
        payload["image_data"] = [encode_image_for_rollout_engine(img) for img in teacher_images]

    teacher_model_name = getattr(args, "teacher_model_name", None)
    teacher_pool_config = getattr(args, "teacher_pool_config", None)
    if teacher_model_name and teacher_pool_config:
        result, endpoint = await get_teacher_pool(args).request_json(payload, request_name="gold_teacher")
        sample.metadata["gold_teacher_endpoint"] = endpoint.name
        sample.metadata["gold_teacher_url"] = endpoint.url
        sample.metadata["gold_teacher_model_name"] = teacher_model_name
    else:
        import aiohttp

        timeout = aiohttp.ClientTimeout(total=600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(args.rm_url, json=payload) as resp:
                resp.raise_for_status()
                result = await resp.json()

    result.setdefault("meta_info", {})["gold_teacher_prompt_len"] = teacher_prompt_len
    result["meta_info"]["gold_teacher_input_ids"] = teacher_input_ids
    return result


async def _call_judge(args, sample: Sample) -> float:
    return await call_llm_judge(args, sample)


async def reward_func_math(args, sample: Sample, **kwargs) -> float:
    """Rule-based math reward for logging/supervision."""
    return compute_math_reward(sample)


async def reward_func_judge(args, sample: Sample, **kwargs) -> float:
    """LLM-judge reward for logging/supervision."""
    return await _call_judge(args, sample)


async def reward_func(args, sample: Sample, **kwargs) -> float:
    """Default GOLD reward function: math raw reward."""
    return await reward_func_math(args, sample, **kwargs)


def _decode_student_response_ids(sample: Sample) -> list[int]:
    return list(sample.tokens[-sample.response_length:]) if sample.response_length > 0 else []


def _to_canonical_pieces_from_ids(tokenizer, token_ids: list[int]) -> list[str]:
    return [tokenizer.decode(ids) for ids in token_ids]


def _to_canonical_pieces_from_triplets(teacher_triplets: list[list]) -> list[str]:
    pieces = []
    prev = ""
    cur = ""
    for item in teacher_triplets:
        piece = str(item[2]) if len(item) > 2 and item[2] is not None else str(item[1])
        cur += piece
        pieces.append(cur[len(prev) :])
        prev = cur
    return pieces


def _extract_teacher_response_triplets(teacher_result: dict, prompt_len: int, response_len: int) -> list[list]:
    triplets = teacher_result["meta_info"]["input_token_logprobs"]
    if response_len <= 0:
        return []

    token_triplets = triplets
    if len(token_triplets) != prompt_len + response_len:
        raise ValueError(
            "Teacher token logprob length does not match local teacher tokenization. "
            f"expected_total={prompt_len + response_len}, actual_total={len(token_triplets)}"
        )

    return token_triplets[prompt_len : prompt_len + response_len]


def _prefix_compatible(a: str, b: str) -> bool:
    return a.startswith(b) or b.startswith(a)


def _find_small_exact_anchor(
    student_pieces: list[str],
    teacher_pieces: list[str],
    student_start: int,
    teacher_start: int,
    cause_side: str | None,
) -> tuple[int, int] | None:
    jump_size = 2
    scan_limit = 5
    max_span = 3
    best_candidate: tuple[int, int] | None = None
    best_score: tuple[int, int, int] | None = None

    def consider_anchor(student_idx: int, teacher_idx: int) -> None:
        nonlocal best_candidate, best_score
        max_student_span = min(max_span, len(student_pieces) - student_idx)
        max_teacher_span = min(max_span, len(teacher_pieces) - teacher_idx)

        for student_span in range(1, max_student_span + 1):
            student_text = "".join(student_pieces[student_idx : student_idx + student_span])
            if not student_text:
                continue
            for teacher_span in range(1, max_teacher_span + 1):
                teacher_text = "".join(teacher_pieces[teacher_idx : teacher_idx + teacher_span])
                if student_text != teacher_text:
                    continue

                matched_chars = len(student_text)
                total_span = student_span + teacher_span
                distance = (student_idx - student_start) + (teacher_idx - teacher_start)
                score = (matched_chars, -total_span, -distance)
                if best_score is None or score > best_score:
                    best_score = score
                    best_candidate = (student_idx, teacher_idx)

    if cause_side == "teacher":
        teacher_anchor = teacher_start + jump_size
        while teacher_anchor < len(teacher_pieces):
            max_student = min(len(student_pieces), student_start + scan_limit)
            for student_idx in range(student_start, max_student):
                consider_anchor(student_idx, teacher_anchor)
            teacher_anchor += 1
    else:
        student_anchor = student_start + jump_size
        while student_anchor < len(student_pieces):
            max_teacher = min(len(teacher_pieces), teacher_start + scan_limit)
            for teacher_idx in range(teacher_start, max_teacher):
                consider_anchor(student_anchor, teacher_idx)
            student_anchor += 1

    return best_candidate


def _find_prefix_anchor(
    student_pieces: list[str],
    teacher_pieces: list[str],
    student_start: int,
    teacher_start: int,
    cause_side: str | None,
) -> tuple[int, int] | None:
    jump_size = 2
    scan_limit = 5

    if cause_side == "teacher":
        teacher_anchor = teacher_start + jump_size
        while teacher_anchor < len(teacher_pieces):
            max_student = min(len(student_pieces), student_start + scan_limit)
            for student_idx in range(student_start, max_student):
                if _prefix_compatible(student_pieces[student_idx], teacher_pieces[teacher_anchor]):
                    return student_idx, teacher_anchor
            teacher_anchor += 1
    else:
        student_anchor = student_start + jump_size
        while student_anchor < len(student_pieces):
            max_teacher = min(len(teacher_pieces), teacher_start + scan_limit)
            for teacher_idx in range(teacher_start, max_teacher):
                if _prefix_compatible(student_pieces[student_anchor], teacher_pieces[teacher_idx]):
                    return student_anchor, teacher_idx
            student_anchor += 1

    return None


def _resync_alignment_start(
    student_pieces: list[str],
    teacher_pieces: list[str],
    student_start: int,
    teacher_start: int,
    cause_side: str | None,
) -> tuple[int, int] | None:
    exact_anchor = _find_small_exact_anchor(
        student_pieces,
        teacher_pieces,
        student_start,
        teacher_start,
        cause_side,
    )
    if exact_anchor is not None:
        return exact_anchor

    return _find_prefix_anchor(
        student_pieces,
        teacher_pieces,
        student_start,
        teacher_start,
        cause_side,
    )


def _build_alignment_groups(student_pieces: list[str], teacher_pieces: list[str]) -> tuple[list[list[int]], list[list[int]]]:
    student_groups: list[list[int]] = []
    teacher_groups: list[list[int]] = []
    student_len = len(student_pieces)
    teacher_len = len(teacher_pieces)
    student_idx = 0
    teacher_idx = 0

    while student_idx < student_len and teacher_idx < teacher_len:

        if student_idx >= student_len or teacher_idx >= teacher_len:
            break

        student_buf = student_pieces[student_idx]
        teacher_buf = teacher_pieces[teacher_idx]
        student_group = [student_idx]
        teacher_group = [teacher_idx]
        current_student_start = student_idx
        current_teacher_start = teacher_idx
        student_idx += 1
        teacher_idx += 1
        last_side: str | None = None
        matched_group = False

        while True:
            if student_buf == teacher_buf:
                student_groups.append(student_group)
                teacher_groups.append(teacher_group)
                matched_group = True
                break

            if not _prefix_compatible(student_buf, teacher_buf):
                break

            if len(student_buf) <= len(teacher_buf):
                if student_idx >= student_len:
                    break
                student_buf += student_pieces[student_idx]
                student_group.append(student_idx)
                student_idx += 1
                last_side = "student"
            else:
                if teacher_idx >= teacher_len:
                    break
                teacher_buf += teacher_pieces[teacher_idx]
                teacher_group.append(teacher_idx)
                teacher_idx += 1
                last_side = "teacher"

        if matched_group:
            continue

        next_start = _resync_alignment_start(
            student_pieces,
            teacher_pieces,
            current_student_start,
            current_teacher_start,
            last_side,
        )
        if next_start is None:
            break
        student_idx, teacher_idx = next_start

    return student_groups, teacher_groups


def _prepare_gold_teacher_log_probs(args, sample: Sample, teacher_result: dict) -> tuple[torch.Tensor, float]:
    mapping_start_time = time.perf_counter()
    student_log_probs = torch.tensor(sample.rollout_log_probs or [], dtype=torch.float32)
    if sample.response_length <= 0:
        sample.metadata["teacher_token_triplets"] = []
        sample.metadata["gold_groups"] = []
        sample.metadata["gold_debug"] = {"student_tokens": [], "teacher_tokens": []}
        return student_log_probs, time.perf_counter() - mapping_start_time

    teacher_prompt_len = int(teacher_result["meta_info"]["gold_teacher_prompt_len"])
    teacher_response_len = len(teacher_result["meta_info"]["gold_teacher_input_ids"]) - teacher_prompt_len
    teacher_triplets = _extract_teacher_response_triplets(teacher_result, teacher_prompt_len, teacher_response_len)
    teacher_log_probs = torch.tensor([float(item[0]) for item in teacher_triplets], dtype=torch.float32)

    student_tokenizer = _get_student_tokenizer(args.hf_checkpoint)
    student_token_ids = _decode_student_response_ids(sample)
    student_pieces = _to_canonical_pieces_from_ids(student_tokenizer, student_token_ids)
    teacher_pieces = _to_canonical_pieces_from_triplets(teacher_triplets)

    student_len = min(len(student_pieces), len(student_log_probs))
    teacher_len = min(len(teacher_pieces), len(teacher_log_probs))
    student_pieces = student_pieces[:student_len]
    student_log_probs = student_log_probs[:student_len]
    teacher_pieces = teacher_pieces[:teacher_len]
    teacher_log_probs = teacher_log_probs[:teacher_len]
    teacher_triplets = teacher_triplets[:teacher_len]

    student_groups, teacher_groups = _build_alignment_groups(student_pieces, teacher_pieces)

    synthetic_teacher_log_probs = student_log_probs.clone()
    gold_groups = []
    for student_group, teacher_group in zip(student_groups, teacher_groups, strict=False):
        if not student_group or not teacher_group:
            continue

        student_group_logprob = sum(float(student_log_probs[idx]) for idx in student_group)
        teacher_group_logprob = sum(float(teacher_log_probs[idx]) for idx in teacher_group)
        group_logprob_gap = student_group_logprob - teacher_group_logprob

        for idx in student_group:
            synthetic_teacher_log_probs[idx] = student_log_probs[idx] - group_logprob_gap

    sample.metadata["teacher_token_triplets"] = teacher_triplets
    mapping_elapsed = time.perf_counter() - mapping_start_time
    print(
        f"[multimodal_gold] sample _prepare_gold_teacher_log_probs total={mapping_elapsed:.6f}s "
        f"student_tokens={len(student_token_ids)} teacher_tokens={len(teacher_triplets)} groups={len(gold_groups)}"
    )
    return synthetic_teacher_log_probs, mapping_elapsed


def _apply_response_prefix_loss_mask(args, sample: Sample) -> None:
    ratio = getattr(args, "gold_train_response_prefix_ratio", None)
    if ratio is None:
        return

    ratio = float(ratio)
    if not (0.0 <= ratio <= 1.0):
        raise ValueError(f"gold_train_response_prefix_ratio must be in [0, 1], got {ratio}")

    response_length = int(sample.response_length)
    if response_length <= 0:
        sample.loss_mask = []
        return

    keep_tokens = math.ceil(response_length * ratio)
    keep_tokens = min(max(keep_tokens, 0), response_length)
    sample.loss_mask = [1] * keep_tokens + [0] * (response_length - keep_tokens)


def _normalize_outcome_rewards(args, raw_scores: list[float]) -> list[float]:
    rewards = torch.tensor(raw_scores, dtype=torch.float32)

    if (
        args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and getattr(args, "rewards_normalization", False)
    ):
        if rewards.shape[-1] == args.n_samples_per_prompt * args.rollout_batch_size:
            rewards = rewards.reshape(-1, args.n_samples_per_prompt)
        else:
            rewards = rewards.view(-1, rewards.shape[-1])

        rewards = rewards - rewards.mean(dim=-1, keepdim=True)

        if args.advantage_estimator in ["grpo", "gspo"] and getattr(args, "grpo_std_normalization", False):
            rewards = rewards / (rewards.std(dim=-1, keepdim=True) + 1e-6)

    return rewards.flatten().tolist()


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Build synthetic teacher log-probs for existing OPD loss path.

    The reward functions return scalar raw rewards. Teacher token log-probs
    (with decoded text pieces for GOLD alignment) are fetched here, similar to
    the multimodal OPD post-process flow.

    Outcome rewards are normalized with the same group-wise GRPO logic as the
    default rollout path, then returned as processed rewards for advantage
    computation.
    """
    import asyncio as _asyncio
    from slime.utils.async_utils import get_async_loop
    raw_scores = [float(sample.reward) for sample in samples]
    # breakpoint()

    async def _gather():
        max_concurrency = max(1, int(getattr(args, "teacher_request_max_concurrency", 32)))
        semaphore = _asyncio.Semaphore(max_concurrency)

        async def _bounded_call(sample: Sample):
            async with semaphore:
                return await _call_teacher(args, sample)

        return await _asyncio.gather(*[_bounded_call(sample) for sample in samples])

    teacher_request_start_time = time.perf_counter()
    teacher_results = get_async_loop().run(_gather())
    teacher_request_elapsed = time.perf_counter() - teacher_request_start_time
    print(f"[multimodal_gold] teacher request finished in {teacher_request_elapsed:.3f}s for {len(samples)} samples")

    mapping_total_elapsed = 0.0
    for sample, teacher_result, raw_reward in zip(samples, teacher_results, raw_scores, strict=False):
        sample.teacher_log_probs, mapping_elapsed = _prepare_gold_teacher_log_probs(args, sample, teacher_result)
        _apply_response_prefix_loss_mask(args, sample)
        mapping_total_elapsed += mapping_elapsed
        sample.metadata["gold_raw_reward"] = raw_reward

    print(
        f"[multimodal_gold] vocab mapping finished in {mapping_total_elapsed:.3f}s total "
        f"({(mapping_total_elapsed / len(samples)) if samples else 0.0:.3f}s/sample)"
    )

    processed_rewards = _normalize_outcome_rewards(args, raw_scores)
    return raw_scores, processed_rewards
