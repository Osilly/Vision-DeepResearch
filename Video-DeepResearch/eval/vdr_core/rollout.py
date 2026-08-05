"""
Custom multi-turn rollout for Vision-DeepResearch tool-calling.

Usage: --custom-generate-function-path examples.vision_deepresearch.rollout.generate

Flow per sample:
    1. Model generates (assistant turn, loss_mask=1)
    2. Env processes: extract <tool_call> or <answer>
       - If <answer>: done, record final answer
       - If <tool_call>: execute tool, encode observation (loss_mask=0)
    3. Repeat until max_turns or model provides <answer>
    4. Finalize: decode response, merge multimodal inputs, prepare teacher GOLD signal, set status
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import aiohttp
import torch

from .slime_utils.processing_utils import (
    encode_image_for_rollout_engine,
    load_processor,
    load_tokenizer,
)
from .slime_utils.types import Sample


class GenerateState:
    """Local shim of slime.rollout.sglang_rollout.GenerateState.

    Only exposes .tokenizer / .processor — the two attributes rollout.generate() reads.
    Process-wide singleton so repeated calls share the loaded tokenizer/processor.
    Note: assumes a single hf_checkpoint per process; a second call with a
    different args.hf_checkpoint returns the first instance's cached values.
    """

    _INSTANCE = None

    def __new__(cls, args):
        if cls._INSTANCE is None:
            inst = super().__new__(cls)
            inst.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
            inst.processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
            cls._INSTANCE = inst
        return cls._INSTANCE

__all__ = ["generate"]

logger = logging.getLogger(__name__)

# Dummy messages for calculating trim length in chat template encoding
DUMMY_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]


def _load_env_module(env_path: str | None):
    target = env_path or "vdr_core.env"
    module_path = Path(target)
    if module_path.suffix == ".py" and module_path.exists():
        spec = importlib.util.spec_from_file_location(f"env_{module_path.stem}", module_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    return importlib.import_module(target)


def _build_env(env_module, sample: Sample, args: Any):
    build_fn = getattr(env_module, "build_env", None)
    if not callable(build_fn):
        raise ValueError("Environment module must expose build_env(sample, args)")
    try:
        return build_fn(sample=sample, args=args)
    except TypeError:
        return build_fn(sample, args)


def _init_trajectory_from_prompt(prompt: Any) -> list[dict[str, Any]]:
    if isinstance(prompt, list):
        return [deepcopy(message) for message in prompt if isinstance(message, dict)]
    return [{"role": "user", "content": prompt}]


def _append_trajectory_message(
    trajectory: list[dict[str, Any]], role: str, content: Any
) -> None:
    trajectory.append({"role": role, "content": content})


def _normalize_observation_for_trajectory(observation: dict[str, Any]) -> dict[str, Any]:
    content = observation.get("obs_str", "")
    message: dict[str, Any] = {"role": "user", "content": content}
    multi_modal_data = observation.get("multi_modal_data") or {}
    if multi_modal_data:
        message["multi_modal_data"] = deepcopy(multi_modal_data)
    return message


def _encode_observation(tokenizer, processor, message: dict, metadata: dict | None, args: Any):
    """Encode observation message for generation and training.

    Returns:
        tuple[prompt_ids, image_data, multimodal_train_inputs]
    """
    tools = metadata.get("tools") if metadata else None
    apply_kwargs = getattr(args, "apply_chat_template_kwargs", None) or {}

    trim_length = 0
    if getattr(args, "apply_chat_template", False):
        dummy = tokenizer.apply_chat_template(DUMMY_MESSAGES, tools=tools, tokenize=False, add_generation_prompt=False, **apply_kwargs)
        formatted = tokenizer.apply_chat_template(DUMMY_MESSAGES + [message], tools=tools, tokenize=False, add_generation_prompt=True, **apply_kwargs)
        trim_length = len(tokenizer.encode(dummy, add_special_tokens=False))
    else:
        formatted = [message]

    multimodal_inputs = None
    multimodal_train_inputs = None
    if processor:
        try:
            from qwen_vl_utils import process_vision_info
            images, _ = process_vision_info([message])
            multimodal_inputs = {"images": images} if images else {}
        except ImportError:
            # Fallback: extract images from message content
            images = []
            content = message.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        images.append(item.get("image"))
                    elif isinstance(item, str) and (item.startswith("http") or os.path.exists(item)):
                        images.append(item)
            multimodal_inputs = {"images": images} if images else {}
        
        if multimodal_inputs.get("images"):
            output = processor(text=formatted, **multimodal_inputs)
            raw_ids = output["input_ids"][0]
            prompt_ids = raw_ids.tolist() if hasattr(raw_ids, "tolist") else list(raw_ids)
            multimodal_train_inputs = {
                k: v for k, v in output.items()
                if k not in ("input_ids", "attention_mask") and not k.startswith("video")
            } or None
        else:
            prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
    else:
        prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)

    if trim_length:
        prompt_ids = prompt_ids[trim_length:]

    image_data = []
    if multimodal_inputs and multimodal_inputs.get("images"):
        image_data = [encode_image_for_rollout_engine(img) for img in multimodal_inputs["images"]]

    return prompt_ids, image_data, multimodal_train_inputs


def _merge_multimodal_inputs(chunks: list[dict | None]) -> dict | None:
    """Merge per-turn multimodal inputs by concatenating tensors."""
    if not chunks:
        return None
    values: dict[str, list] = {}
    for chunk in chunks:
        if not chunk:
            continue
        for k, v in chunk.items():
            if k.startswith("video"):
                continue
            if v is not None and isinstance(v, torch.Tensor) and v.numel() > 0:
                values.setdefault(k, []).append(v)
    if not values:
        return None
    result = {}
    for k, v in values.items():
        merged = torch.cat(v, dim=0)
        if merged.numel() > 0:
            result[k] = merged
    return result if result else None


import os

_HTTP_SESSION: aiohttp.ClientSession | None = None


def _get_http_session() -> aiohttp.ClientSession:
    global _HTTP_SESSION
    if _HTTP_SESSION is None or _HTTP_SESSION.closed:
        _HTTP_SESSION = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=12000))
    return _HTTP_SESSION


async def _run_inference(url: str, tokens: list[int], sampling_params: dict, image_data, *, text: str | None = None):
    """Call SGLang for one generation turn.

    For the first multimodal turn pass ``text`` (the formatted prompt string) so
    SGLang tokenises and expands image tokens internally.  Subsequent turns use
    ``input_ids`` (compressed tokeniser tokens + accumulated response/obs tokens).
    """
    payload = {"sampling_params": sampling_params, "return_logprob": True}
    if text is not None:
        payload["text"] = text
    else:
        payload["input_ids"] = tokens
    if image_data:
        payload["image_data"] = image_data

    session = _get_http_session()
    async with session.post(url, json=payload) as resp:
        if resp.status != 200:
            body = await resp.text()
            logger.error("SGLang /generate returned %s: %s", resp.status, body[:500])
        resp.raise_for_status()
        output = await resp.json()

    response_text = output["text"]
    meta = output.get("meta_info", {})

    if "output_token_logprobs" in meta:
        new_tokens = [item[1] for item in meta["output_token_logprobs"]]
        new_logprobs = [item[0] for item in meta["output_token_logprobs"]]
    else:
        new_tokens, new_logprobs = [], []

    finish_type = meta.get("finish_reason", {}).get("type", "")
    return response_text, new_tokens, new_logprobs, finish_type


def _prepare_initial_multimodal_inputs(sample: Sample, processor, prompt):
    """Prepare initial multimodal inputs for the first turn."""
    if not sample.multimodal_inputs:
        return None, None
    
    mm_inputs = {k: v for k, v in sample.multimodal_inputs.items() if not k.startswith("video")}
    if not mm_inputs:
        return None, None
    
    try:
        proc_output = processor(text=prompt, **mm_inputs)
        raw_ids = proc_output["input_ids"][0]
        prompt_ids = raw_ids.tolist() if hasattr(raw_ids, "tolist") else list(raw_ids)
        mm_train_inputs = {
            k: v for k, v in proc_output.items()
            if k not in ("input_ids", "attention_mask") and not k.startswith("video")
        } or None
        return prompt_ids, mm_train_inputs
    except Exception:
        logger.exception("_prepare_initial_multimodal_inputs failed, falling back to text-only")
        return None, None


def _prepare_initial_image_data(sample: Sample) -> list[str]:
    """Prepare initial image data from sample.

    Primary path: use pre-loaded PIL images from sample.multimodal_inputs.
    Lazy path: when multimodal_inputs is absent (skipped at build time for
    speed), extract image file paths from the raw_prompt and load them here,
    inside the per-sample rollout coroutine so I/O is naturally concurrent.
    """
    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        images = sample.multimodal_inputs["images"]
        return [encode_image_for_rollout_engine(img) for img in images]

    raw_prompt = (sample.metadata or {}).get("raw_prompt")
    if not raw_prompt:
        return []

    from PIL import Image as _PILImage
    results = []
    for msg in raw_prompt:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        for item in (msg.get("content") or []):
            if isinstance(item, dict) and item.get("type") == "image":
                path = item.get("image")
                if path:
                    try:
                        img = _PILImage.open(path).convert("RGB")
                        results.append(encode_image_for_rollout_engine(img))
                    except Exception:
                        logger.warning("failed to load initial image: %s", path)
        break  # only the first user message contains the initial video frames
    return results


def _append_tokens(
    sample: Sample,
    train_response_tokens: list[int],
    all_response_tokens: list[int],
    assistant_positions: list[int],
    tokens: list[int],
    logprobs: list[float],
    loss_mask: int,
):
    """Append tokens to sample with specified loss_mask."""
    start = len(all_response_tokens)
    sample.tokens.extend(tokens)
    all_response_tokens.extend(tokens)
    sample.loss_mask.extend([loss_mask] * len(tokens))
    sample.rollout_log_probs.extend(logprobs)
    if loss_mask == 1:
        train_response_tokens.extend(tokens)
        assistant_positions.extend(range(start, start + len(tokens)))


async def generate(args: Any, sample: Sample, sampling_params, evaluation: bool = False) -> Sample:
    """
    Main entry point for multi-turn Vision-DeepResearch tool-calling rollout.
    
    Key design:
    - Model outputs (assistant turns) have loss_mask=1 → participate in training
    - Tool observations have loss_mask=0 → do not participate in training
    """
    assert not args.partial_rollout, "Partial rollout not supported"

    # Inject tools schema
    from .tools.registry import get_tools
    sample.metadata = sample.metadata or {}
    sample.metadata.setdefault("tools", get_tools())

    # Init env and state
    env_module = _load_env_module(getattr(args, "rollout_interaction_env_path", None))
    max_turns = args.max_turns
    if max_turns is None:
        raise ValueError("max_turns must be set via --custom-config-path")

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    env = _build_env(env_module, sample, args)

    # For phased/adaptive mode, restrict env to phase1 tools so that what
    # the env will execute matches what the model sees in its system prompt.
    _phase1_tool_names = sample.metadata.get("phase1_tool_names") if sample.metadata else None
    _phase2_tool_names = sample.metadata.get("phase2_tool_names") if sample.metadata else None
    if _phase1_tool_names and hasattr(env, "supported_tool_names"):
        env.supported_tool_names = list(_phase1_tool_names)

    # === Gym-style reset: env builds the (system, first user observation) pair
    # at runtime so {date} stays fresh and the tool list comes straight from
    # tools/registry.py. Overrides whatever slime's data pipeline rendered.
    user_obs_msg, _reset_info, system_message = await env.reset_async(sample)
    prompt_messages = [
        {"role": "system", "content": system_message},
        user_obs_msg,
    ]
    sample.metadata["raw_prompt"] = prompt_messages
    # env.question is set by env.reset() — used for terminal-reward judge prompt
    sample.metadata["question"] = getattr(env, "question", "")

    apply_kwargs = getattr(args, "apply_chat_template_kwargs", None) or {}
    # Render the compressed (one image-pad per image) form for sample.tokens.
    # We deliberately do NOT pass tools=... — the tool schema is already inlined
    # into the system content via {tools}, mirroring eval. Passing tools again
    # would double the <tools> block.
    rendered_prompt = await asyncio.to_thread(
        state.tokenizer.apply_chat_template,
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        **apply_kwargs,
    )
    sample.prompt = rendered_prompt

    # Encode to compressed token ids (one image-pad token per image, matching what
    # sglang expects alongside image_data). Use processor only for mm_train_inputs.
    prompt_ids = await asyncio.to_thread(
        state.tokenizer.encode, rendered_prompt, add_special_tokens=False
    )
    sample.tokens = list(prompt_ids)
    sample.loss_mask = sample.loss_mask or []
    sample.rollout_log_probs = sample.rollout_log_probs or []

    mm_train_inputs = None
    if sample.multimodal_inputs and state.processor:
        _, mm_train_inputs = await asyncio.to_thread(
            _prepare_initial_multimodal_inputs, sample, state.processor, rendered_prompt
        )

    # Prepare initial images for sglang's image_data payload.
    current_images = []
    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        current_images = await asyncio.to_thread(_prepare_initial_image_data, sample)
        logger.info("multimodal prompt: %d images", len(current_images))

    train_response_tokens: list[int] = []
    all_response_tokens: list[int] = []
    assistant_positions: list[int] = []
    trajectory = _init_trajectory_from_prompt(prompt_messages)
    mm_inputs_buffer: list = [mm_train_inputs] if mm_train_inputs else []
    budget = None
    if args.rollout_max_context_len:
        budget = args.rollout_max_context_len
    elif sampling_params.get("max_new_tokens"):
        budget = sampling_params["max_new_tokens"]

    max_tokens_per_turn = getattr(args, "max_tokens_per_turn", None)
    if max_tokens_per_turn is not None:
        max_tokens_per_turn = int(max_tokens_per_turn)
        if max_tokens_per_turn <= 0:
            raise ValueError("max_tokens_per_turn must be positive when set")

    sampling_params = deepcopy(sampling_params)
    sample.status = None

    # === Env-owned reward accumulator. Set on sample.reward at the end so
    # slime's `if sample.reward is None` guard at sglang_rollout.py:277 skips
    # the external RM call (we don't need --custom-rm-path anymore).
    total_reward = 0.0
    response_text = ""   # last assistant text; "" until first turn finishes
    env_emitted_done = False  # True when env.step returned done with a judge reward

    try:

        force_answer_on_last_turn = bool(
            getattr(args, "force_answer_on_last_turn", False)
        )

        # === Phased rollout: read transition payload set by the producer ===
        # The async data producer (phased mode) stashes:
        #   metadata["phase_switch_turn"]:   int, the turn at which to switch
        #   metadata["phase_transition_text"]: synthetic user message that
        #                                      re-declares the full toolset.
        # Single-mode rollouts have neither, so the injection is a no-op.
        phase_switch_turn = (
            sample.metadata.get("phase_switch_turn") if sample.metadata else None
        )
        phase_transition_text = (
            sample.metadata.get("phase_transition_text") if sample.metadata else None
        )

        # === Adaptive rollout state ===
        rollout_mode = sample.metadata.get("rollout_mode", "single") if sample.metadata else "single"
        adaptive_phase1_force_turns = sample.metadata.get("adaptive_phase1_force_turns", 0) if sample.metadata else 0
        adaptive_phase1_max_turns = sample.metadata.get("adaptive_phase1_max_turns", 0) if sample.metadata else 0
        visual_end_marker = sample.metadata.get("visual_end_marker", "[Visual exploration complete]") if sample.metadata else "[Visual exploration complete]"
        adaptive_transitioned = False   # have we injected the full-tool transition?
        need_adaptive_transition = False  # marker detected last turn → inject this turn

        for turn in range(max_turns):
            # Check budget
            if budget is not None and budget <= 0:
                sample.status = Sample.Status.TRUNCATED
                break

            # === Step 0a: Phase 1 → Phase 2 switch (phased / adaptive) ===
            # Phased mode: inject at the fixed boundary turn.
            # Adaptive mode: inject when model emitted the end marker (previous
            # turn) OR when the max visual-phase turn is reached.
            _inject_phase_transition = False
            if phase_transition_text:
                if rollout_mode == "adaptive" and not adaptive_transitioned:
                    if need_adaptive_transition or (
                        adaptive_phase1_max_turns > 0 and turn == adaptive_phase1_max_turns
                    ):
                        _inject_phase_transition = True
                        adaptive_transitioned = True
                        need_adaptive_transition = False
                        logger.info("[adaptive] turn %d: injecting full-tool transition", turn)
                elif rollout_mode != "adaptive" and phase_switch_turn is not None and turn == phase_switch_turn:
                    _inject_phase_transition = True

            if _inject_phase_transition:
                transition_msg = {
                    "role": "user",
                    "content": [{"type": "text", "text": phase_transition_text}],
                }
                _append_trajectory_message(trajectory, "user", phase_transition_text)
                t_ids, _t_images, t_mm_train = await asyncio.to_thread(
                    _encode_observation,
                    state.tokenizer, state.processor, transition_msg,
                    sample.metadata, args,
                )
                if (
                    state.tokenizer.bos_token_id
                    and t_ids
                    and t_ids[0] == state.tokenizer.bos_token_id
                ):
                    t_ids = t_ids[1:]
                _append_tokens(
                    sample,
                    train_response_tokens,
                    all_response_tokens,
                    assistant_positions,
                    t_ids,
                    [0.0] * len(t_ids),
                    loss_mask=0,
                )
                if budget is not None:
                    budget -= len(t_ids)
                mm_inputs_buffer.append(t_mm_train)
                # Switch env to phase2 tools so execution matches the new prompt.
                if _phase2_tool_names and hasattr(env, "supported_tool_names"):
                    env.supported_tool_names = list(_phase2_tool_names)

            # === Step 0: Final-turn forcing (rollout-only opt-in) ===
            # On the last allowed turn, push a synthetic user message telling
            # the model: no more tool calls — produce <answer>...</answer> now
            # from whatever evidence it has accumulated. Salvages an answer
            # instead of ending the rollout on an unanswered tool response.
            # Gated by args.force_answer_on_last_turn so evaluation runs are
            # unaffected.
            if (
                force_answer_on_last_turn
                and max_turns > 1
                and turn == max_turns - 1
            ):
                force_text = (
                    "You have reached the final allowed turn. Do NOT call any more tools. "
                    "Based on all the information gathered above, write your best final answer now, "
                    "wrapped in <answer></answer> tags."
                )
                force_msg = {
                    "role": "user",
                    "content": [{"type": "text", "text": force_text}],
                }
                _append_trajectory_message(trajectory, "user", force_text)
                force_ids, _force_images, force_mm_train = await asyncio.to_thread(
                    _encode_observation,
                    state.tokenizer, state.processor, force_msg,
                    sample.metadata, args,
                )
                if (
                    state.tokenizer.bos_token_id
                    and force_ids
                    and force_ids[0] == state.tokenizer.bos_token_id
                ):
                    force_ids = force_ids[1:]
                _append_tokens(
                    sample,
                    train_response_tokens,
                    all_response_tokens,
                    assistant_positions,
                    force_ids,
                    [0.0] * len(force_ids),
                    loss_mask=0,
                )
                if budget is not None:
                    budget -= len(force_ids)
                mm_inputs_buffer.append(force_mm_train)

            # === Step 1: Model Generation (loss_mask=1) ===
            turn_max_new_tokens = budget
            if max_tokens_per_turn is not None:
                turn_max_new_tokens = (
                    min(budget, max_tokens_per_turn)
                    if budget is not None
                    else max_tokens_per_turn
                )
            cur_params = (
                {**sampling_params, "max_new_tokens": turn_max_new_tokens}
                if turn_max_new_tokens is not None
                else sampling_params
            )
            # First turn with images: send text so SGLang handles image token expansion
            # internally.  Subsequent turns use accumulated input_ids.
            use_text = (turn == 0 and current_images and isinstance(sample.prompt, str))
            response_text, new_tokens, new_logprobs, finish_type = await _run_inference(
                url, sample.tokens, cur_params, current_images,
                text=sample.prompt if use_text else None,
            )
            _append_trajectory_message(trajectory, "assistant", response_text)
            
            _append_tokens(
                sample,
                train_response_tokens,
                all_response_tokens,
                assistant_positions,
                new_tokens,
                new_logprobs,
                loss_mask=1,  # Model outputs participate in training
            )
            if budget is not None:
                budget -= len(new_tokens)

            # Check finish type
            if finish_type in ("length", "abort"):
                sample.status = Sample.Status.TRUNCATED if finish_type == "length" else Sample.Status.ABORTED
                break

            # === Step 2: Environment Processing (env owns reward) ===
            observation, step_reward, done, info = await env.step_async(response_text)
            total_reward += float(step_reward)
            trajectory_observation = _normalize_observation_for_trajectory(observation)
            if trajectory_observation["content"]:
                trajectory.append(trajectory_observation)

            if done:
                # Env emitted a natural done (saw <answer> and called the judge,
                # or hit consecutive-error / is_final_turn). Reward is already
                # in `total_reward`; do not call compute_final_reward.
                env_emitted_done = True
                sample.status = Sample.Status.COMPLETED
                break

            # === Adaptive: detect visual end marker (optional phase only) ===
            # Turns [0, adaptive_phase1_force_turns) are forced visual;
            # starting from adaptive_phase1_force_turns the model may signal
            # "视觉探索部分已结束" to voluntarily end the visual phase.
            if (
                rollout_mode == "adaptive"
                and not adaptive_transitioned
                and not need_adaptive_transition
                and turn >= adaptive_phase1_force_turns
                and visual_end_marker in response_text
            ):
                need_adaptive_transition = True
                logger.info("[adaptive] turn %d: visual end marker detected, will transition next turn", turn)

            # === Step 3: Encode Observation (loss_mask=0) ===
            next_msg = env.format_observation(observation)
            obs_ids, obs_images, obs_mm_train_inputs = await asyncio.to_thread(
                _encode_observation,
                state.tokenizer,
                state.processor,
                next_msg,
                sample.metadata,
                args,
            )
            
            # Remove BOS token if present
            if state.tokenizer.bos_token_id and obs_ids and obs_ids[0] == state.tokenizer.bos_token_id:
                obs_ids = obs_ids[1:]

            _append_tokens(
                sample,
                train_response_tokens,
                all_response_tokens,
                assistant_positions,
                obs_ids,
                [0.0] * len(obs_ids),
                loss_mask=0,  # Tool observations do NOT participate in training
            )
            if budget is not None:
                budget -= len(obs_ids)

            # === Step 4: Accumulate Images ===
            if obs_images:
                current_images.extend(obs_images)
            if observation.get("multi_modal_data"):
                sample.multimodal_inputs = sample.multimodal_inputs or {}
                for k, v in observation["multi_modal_data"].items():
                    if v:
                        sample.multimodal_inputs.setdefault(k, []).extend(v)
            mm_inputs_buffer.append(obs_mm_train_inputs)

            # === Step 5: Check Max Turns ===
            if turn + 1 >= max_turns:
                sample.status = Sample.Status.COMPLETED
                break

        # === Finalize ===
        if sample.status is None:
            sample.status = Sample.Status.COMPLETED

        # Merge multimodal inputs
        sample.multimodal_train_inputs = _merge_multimodal_inputs(mm_inputs_buffer)

        # Decode response (only assistant tokens)
        sample.response = state.tokenizer.decode(train_response_tokens, skip_special_tokens=False)
        sample.response_length = len(all_response_tokens)

        # === Terminal reward for non-natural exits ===
        # If the env didn't emit done=True (i.e. loop broke via budget /
        # finish_type length|abort / max_turns hit on a tool turn), judge the
        # last assistant text so a model that ran out of room still gets
        # credit for the best partial answer it produced.
        if not env_emitted_done and response_text:
            try:
                terminal_reward = await env.compute_final_reward(response_text)
                total_reward += float(terminal_reward)
            except Exception as exc:
                logger.warning("compute_final_reward failed: %s", exc)

        # Hand reward to slime. The check at sglang_rollout.py:277
        # (`if sample.reward is None`) then skips the external RM path.
        sample.reward = float(total_reward)

        # Store assistant positions and full trajectory for debugging/analysis
        sample.metadata["student_assistant_response_positions"] = assistant_positions
        sample.metadata["trajectory"] = trajectory
        sample.metadata["conversation"] = trajectory

        return sample

    finally:
        try:
            await env.close()
        except Exception:
            pass

