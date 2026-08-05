"""
Multimodal OPD/GRPO reward and post-process functions.

Design: reward_func and post_process_rewards are decoupled so that
any reward can be combined with any training algorithm via script args:

  Reward functions (--custom-rm-path):
    slime.rollout.multimodal_opd.reward_func_math    - rule-based math grader
    slime.rollout.multimodal_opd.reward_func_judge   - vLLM LLM-judge
    slime.rollout.multimodal_opd.reward_func         - alias for reward_func_math

  Post-process functions (--custom-reward-post-process-path):
    slime.rollout.multimodal_opd.post_process_grpo        - group-norm only (pure GRPO)
    slime.rollout.multimodal_opd.post_process_opd         - group-norm + OPD teacher KL
    slime.rollout.multimodal_opd.post_process_opd_kl_only - keep raw judge rewards for logging, use only KL as advantage
    slime.rollout.multimodal_opd.post_process_rewards     - alias for post_process_opd

  Combinations:

    # GRPO + math reward
    --custom-rm-path slime.rollout.multimodal_opd.reward_func_math
    --custom-reward-post-process-path slime.rollout.multimodal_opd.post_process_grpo

    # GRPO + LLM-judge reward
    --custom-rm-path slime.rollout.multimodal_opd.reward_func_judge
    --custom-reward-post-process-path slime.rollout.multimodal_opd.post_process_grpo

    # OPD + math reward
    --custom-rm-path slime.rollout.multimodal_opd.reward_func_math
    --custom-reward-post-process-path slime.rollout.multimodal_opd.post_process_opd

    # OPD + LLM-judge reward
    --custom-rm-path slime.rollout.multimodal_opd.reward_func_judge
    --custom-reward-post-process-path slime.rollout.multimodal_opd.post_process_opd

    # OPD KL-only + LLM-judge logging
    --custom-rm-path slime.rollout.multimodal_opd.reward_func_judge
    --custom-reward-post-process-path slime.rollout.multimodal_opd.post_process_opd_kl_only
"""
import logging

import torch

from slime.rollout.rm_hub.multimodal import call_llm_judge, call_multimodal_teacher, compute_math_reward
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _call_teacher(args, sample: Sample) -> dict:
    """Call SGLang teacher server for token-level log-probs (OPD signal).

    Context mode via env var USE_TEACHER_CONTEXT:
      0 (default): symmetric - teacher uses same tokens as student.
      1: asymmetric - teacher uses sample.metadata["teacher_prompt_ids"]
         + student response tokens.
    """
    return await call_multimodal_teacher(
        args,
        sample,
        return_text_in_logprobs=False,
        use_teacher_context_env=True,
    )


async def _call_judge(args, sample: Sample) -> float:
    """Call vLLM judge server to score correctness.

    Required env vars:
        JUDGE_URL   e.g. http://10.x.x.x:8000/v1
        JUDGE_MODEL model name served by vLLM (default: "default")
    """
    return await call_llm_judge(args, sample)



def _group_normalize(
    scores: list[float],
    n_samples: int,
    grpo_std_normalization: bool,
) -> list[float]:
    """Group-wise reward normalization: subtract group mean, optionally divide by std."""
    rewards = torch.tensor(scores, dtype=torch.float32)
    if rewards.numel() % n_samples == 0:
        g = rewards.view(-1, n_samples)
        normalized = g - g.mean(dim=1, keepdim=True)
        if grpo_std_normalization:
            normalized = normalized / (g.std(dim=1, keepdim=True) + 1e-6)
        return normalized.view(-1).tolist()
    else:
        logger.warning(
            f"Sample count {rewards.numel()} not divisible by "
            f"n_samples_per_prompt {n_samples}, skipping group normalization."
        )
        return scores


def _attach_reward_metrics(samples, scores, teacher_log_probs=None):
    """Store per-sample metrics in metadata for custom_rollout_log aggregation."""
    for i, (sample, ms) in enumerate(zip(samples, scores)):
        m = {"math_correct": float(ms)}

        if teacher_log_probs is not None:
            t_lp = teacher_log_probs[i]
            if len(t_lp) > 0:
                m["teacher_logprob_mean"] = t_lp.mean().item()
            if sample.rollout_log_probs:
                s_lp = torch.tensor(sample.rollout_log_probs, dtype=torch.float32)
                m["student_logprob_mean"] = s_lp.mean().item()
                min_len = min(len(t_lp), len(s_lp))
                if min_len > 0:
                    gap = (t_lp[:min_len] - s_lp[:min_len]).mean().item()
                    m["logprob_gap"] = gap
                    pt = t_lp[:min_len].exp().clamp(max=1.0)
                    m["kl_teacher_student"] = (
                        pt * (t_lp[:min_len] - s_lp[:min_len])
                    ).mean().item()

        sample.metadata["_opd_metrics"] = m


# ---------------------------------------------------------------------------
# Reward functions  (--custom-rm-path)
# ---------------------------------------------------------------------------

async def reward_func_math(args, sample: Sample, **kwargs) -> float:
    """Rule-based math reward. Returns 1.0 if correct, else 0.0.

    Safe during eval rollouts (no teacher/judge calls).
    """
    return compute_math_reward(sample)


async def reward_func_judge(args, sample: Sample, **kwargs) -> float:
    """LLM-judge reward via vLLM. Returns 1.0 if judge says correct.

    Requires env vars: JUDGE_URL, JUDGE_MODEL.
    Safe during eval rollouts.
    """
    return await _call_judge(args, sample)


# ---------------------------------------------------------------------------
# Post-process functions  (--custom-reward-post-process-path)
# ---------------------------------------------------------------------------

def post_process_grpo(args, samples: list[Sample], **kwargs):
    """Pure GRPO: group-normalize rewards only. No teacher call.

    Use with: --advantage-estimator grpo
    Compatible with any reward_func_*.
    """
    scores = [float(sample.reward) for sample in samples]
    _attach_reward_metrics(samples, scores, teacher_log_probs=None)

    normalized = _group_normalize(
        scores,
        n_samples=args.n_samples_per_prompt,
        grpo_std_normalization=getattr(args, "grpo_std_normalization", False),
    )
    return scores, normalized


def post_process_opd(args, samples: list[Sample], **kwargs):
    """GRPO + OPD: group-normalize rewards AND fetch teacher log-probs for KL.

    Use with: --advantage-estimator grpo --use-opd --opd-kl-coef <coef>
              --rm-url http://<teacher-ip>:<port>/generate
    Compatible with any reward_func_*.
    """
    import asyncio as _asyncio
    from slime.utils.async_utils import get_async_loop

    response_lengths = [sample.response_length for sample in samples]
    scores = [float(sample.reward) for sample in samples]

    # 1. Fetch teacher log-probs concurrently
    async def _gather():
        return await _asyncio.gather(*[_call_teacher(args, s) for s in samples])

    teacher_results = get_async_loop().run(_gather())

    # 2. Trim to response segment
    teacher_log_probs = [
        torch.tensor(
            [item[0] for item in r["meta_info"]["input_token_logprobs"][1:]],
            dtype=torch.float32,
        )[-resp_len:]
        for r, resp_len in zip(teacher_results, response_lengths)
    ]
    for sample, t_lp in zip(samples, teacher_log_probs):
        sample.teacher_log_probs = t_lp

    # 3. Attach metrics for logging
    _attach_reward_metrics(samples, scores, teacher_log_probs=teacher_log_probs)

    # 4. Group-normalize
    normalized = _group_normalize(
        scores,
        n_samples=args.n_samples_per_prompt,
        grpo_std_normalization=getattr(args, "grpo_std_normalization", False),
    )
    return scores, normalized


def post_process_opd_kl_only(args, samples: list[Sample], **kwargs):
    """OPD KL-only: keep raw rewards for logging, but zero out reward advantages.

    Use with: --advantage-estimator grpo --use-opd --opd-kl-coef <coef>
              --rm-url http://<teacher-ip>:<port>/generate
    Compatible with any reward_func_*.

    Returns:
        raw rewards: original sample.reward values for logging/metrics.
        processed rewards: all-zero rewards so the final advantage signal comes
        only from OPD KL added in compute_advantages_and_returns().
    """
    import asyncio as _asyncio
    from slime.utils.async_utils import get_async_loop

    response_lengths = [sample.response_length for sample in samples]
    scores = [float(sample.reward) for sample in samples]

    async def _gather():
        return await _asyncio.gather(*[_call_teacher(args, s) for s in samples])

    teacher_results = get_async_loop().run(_gather())

    teacher_log_probs = [
        torch.tensor(
            [item[0] for item in r["meta_info"]["input_token_logprobs"][1:]],
            dtype=torch.float32,
        )[-resp_len:]
        for r, resp_len in zip(teacher_results, response_lengths)
    ]
    for sample, t_lp in zip(samples, teacher_log_probs):
        sample.teacher_log_probs = t_lp

    _attach_reward_metrics(samples, scores, teacher_log_probs=teacher_log_probs)

    zero_rewards = [0.0] * len(samples)
    return scores, zero_rewards



