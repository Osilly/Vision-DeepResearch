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

import os
import time

import aiohttp
import torch
from transformers import AutoTokenizer

from slime.rollout.rm_hub.multimodal import call_llm_judge, compute_math_reward
from slime.utils.processing_utils import build_processor_kwargs, encode_image_for_rollout_engine, load_processor
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
        enable_thinking=False,
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


def _build_alignment_groups(student_pieces: list[str], teacher_pieces: list[str]) -> tuple[list[list[int]], list[list[int]]]:
    i = j = 0
    student_buf = teacher_buf = ""
    student_group: list[int] = []
    teacher_group: list[int] = []
    student_groups: list[list[int]] = []
    teacher_groups: list[list[int]] = []

    def flush_group():
        nonlocal student_buf, teacher_buf, student_group, teacher_group
        if student_group and teacher_group:
            student_groups.append(student_group.copy())
            teacher_groups.append(teacher_group.copy())
        student_buf = teacher_buf = ""
        student_group = []
        teacher_group = []

    while i < len(student_pieces) or j < len(teacher_pieces):
        if student_buf == teacher_buf and student_buf != "":
            flush_group()
            continue

        if student_buf == "" and i < len(student_pieces):
            student_buf += student_pieces[i]
            student_group.append(i)
            i += 1
            continue

        if teacher_buf == "" and j < len(teacher_pieces):
            teacher_buf += teacher_pieces[j]
            teacher_group.append(j)
            j += 1
            continue

        if len(student_buf) <= len(teacher_buf):
            if i < len(student_pieces):
                student_buf += student_pieces[i]
                student_group.append(i)
                i += 1
            elif j < len(teacher_pieces):
                teacher_buf += teacher_pieces[j]
                teacher_group.append(j)
                j += 1
        else:
            if j < len(teacher_pieces):
                teacher_buf += teacher_pieces[j]
                teacher_group.append(j)
                j += 1
            elif i < len(student_pieces):
                student_buf += student_pieces[i]
                student_group.append(i)
                i += 1

    if student_buf == teacher_buf and student_group and teacher_group:
        flush_group()
    elif student_group or teacher_group:
        student_groups.append(student_group.copy() if student_group else [])
        teacher_groups.append(teacher_group.copy() if teacher_group else [])

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
    teacher_tokens = [str(item[2]) if len(item) > 2 and item[2] is not None else str(item[1]) for item in teacher_triplets]

    student_tokenizer = _get_student_tokenizer(args.hf_checkpoint)
    student_token_ids = _decode_student_response_ids(sample)
    student_pieces_start_time = time.perf_counter()
    student_pieces = _to_canonical_pieces_from_ids(student_tokenizer, student_token_ids)
    student_pieces_elapsed = time.perf_counter() - student_pieces_start_time
    teacher_pieces_start_time = time.perf_counter()
    teacher_pieces = _to_canonical_pieces_from_triplets(teacher_triplets)
    teacher_pieces_elapsed = time.perf_counter() - teacher_pieces_start_time
    print(
        f"[multimodal_gold] piece build student={student_pieces_elapsed:.6f}s teacher={teacher_pieces_elapsed:.6f}s "
        f"student_tokens={len(student_token_ids)} teacher_tokens={len(teacher_triplets)}"
    )

    student_len = min(len(student_pieces), len(student_log_probs))
    teacher_len = min(len(teacher_pieces), len(teacher_log_probs))
    student_pieces = student_pieces[:student_len]
    student_log_probs = student_log_probs[:student_len]
    teacher_pieces = teacher_pieces[:teacher_len]
    teacher_log_probs = teacher_log_probs[:teacher_len]
    teacher_triplets = teacher_triplets[:teacher_len]
    teacher_tokens = teacher_tokens[:teacher_len]

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

        gold_groups.append(
            {
                "student_group": student_group,
                "teacher_group": teacher_group,
                "student_text": "".join(student_pieces[idx] for idx in student_group),
                "teacher_text": "".join(teacher_pieces[idx] for idx in teacher_group),
                "student_group_logprob": student_group_logprob,
                "teacher_group_logprob": teacher_group_logprob,
                "group_logprob_gap": group_logprob_gap,
            }
        )

    sample.metadata["teacher_token_triplets"] = teacher_triplets
    sample.metadata["gold_groups"] = gold_groups
    sample.metadata["gold_debug"] = {
        "student_tokens": student_pieces,
        "teacher_tokens": teacher_pieces,
        "student_raw_tokens": [str(tok) for tok in student_tokenizer.convert_ids_to_tokens(student_token_ids[:student_len])],
        "teacher_raw_tokens": teacher_tokens,
        "group_mappings": [
            {
                "student_group": item["student_group"],
                "teacher_group": item["teacher_group"],
                "student_text": item["student_text"],
                "teacher_text": item["teacher_text"],
                "group_logprob_gap": item["group_logprob_gap"],
            }
            for item in gold_groups
        ],
    }
    mapping_elapsed = time.perf_counter() - mapping_start_time
    print(
        f"[multimodal_gold] sample _prepare_gold_teacher_log_probs total={mapping_elapsed:.6f}s "
        f"student_tokens={len(student_token_ids)} teacher_tokens={len(teacher_triplets)} groups={len(gold_groups)}"
    )
    return synthetic_teacher_log_probs, mapping_elapsed


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Build synthetic teacher log-probs for existing OPD loss path.

    The reward functions return scalar raw rewards. Teacher token log-probs
    (with decoded text pieces for GOLD alignment) are fetched here, similar to
    the multimodal OPD post-process flow.

    No scalar reward is used for training in this algorithm. We always return:
    - raw rewards: for logging / supervision
    - processed rewards: all zeros so advantage comes only from GOLD-projected
      teacher log-probs.
    """
    import asyncio as _asyncio
    from slime.utils.async_utils import get_async_loop
    raw_scores = [float(sample.reward) for sample in samples]
    breakpoint()

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
        mapping_total_elapsed += mapping_elapsed
        sample.metadata["gold_raw_reward"] = raw_reward

    print(
        f"[multimodal_gold] vocab mapping finished in {mapping_total_elapsed:.3f}s total "
        f"({(mapping_total_elapsed / len(samples)) if samples else 0.0:.3f}s/sample)"
    )

    zero_rewards = [0.0] * len(samples)
    return raw_scores, zero_rewards

