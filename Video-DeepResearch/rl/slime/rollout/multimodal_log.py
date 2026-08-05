"""
Custom rollout and eval log functions for multimodal OPD training.

Wired via:
  --custom-rollout-log-function-path slime.rollout.multimodal_log.custom_rollout_log
  --custom-eval-rollout-log-function-path slime.rollout.multimodal_log.custom_eval_log
"""
import logging

import numpy as np

from slime.utils.logging_utils import log
from slime.utils.metric_utils import (
    compute_rollout_step,
    dict_add_prefix,
)

logger = logging.getLogger(__name__)


def _collect_opd_metrics(samples) -> dict:
    """Aggregate per-sample _opd_metrics stored by post_process_rewards."""
    opd_list = [
        s.metadata["_opd_metrics"]
        for s in samples
        if isinstance(s.metadata, dict) and "_opd_metrics" in s.metadata
    ]
    if not opd_list:
        return {}

    def _mean(key):
        vals = [m[key] for m in opd_list if key in m]
        return float(np.mean(vals)) if vals else None

    result = {}
    for key in (
        "math_correct",
        "teacher_logprob_mean",
        "student_logprob_mean",
        "logprob_gap",
        "kl_teacher_student",
    ):
        v = _mean(key)
        if v is not None:
            result[key] = v
    return result


def custom_rollout_log(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    """Custom rollout metrics logger.

    Adds on top of the default metrics:
      rollout/judge_accuracy          - fraction of correct responses in this batch
      rollout/opd/teacher_logprob_mean
      rollout/opd/student_logprob_mean
      rollout/opd/logprob_gap         - teacher minus student logprob (token avg)
      rollout/opd/kl_teacher_student  - approximate KL(teacher||student)

    Returns True to signal the framework that logging is handled here.
    """
    log_dict = {**(rollout_extra_metrics or {})}

    # Standard framework metrics (response length, zero_std, truncated_ratio, etc.)
    from slime.ray.rollout import compute_metrics_from_samples, compute_perf_metrics_from_samples
    log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), "rollout/")
    log_dict |= dict_add_prefix(
        compute_perf_metrics_from_samples(args, samples, rollout_time), "perf/"
    )

    # OPD + math reward metrics from metadata
    opd = _collect_opd_metrics(samples)
    if opd:
        if "math_correct" in opd:
            log_dict["rollout/math_accuracy"] = opd["math_correct"]
        for key in ("teacher_logprob_mean", "student_logprob_mean", "logprob_gap", "kl_teacher_student"):
            if key in opd:
                log_dict[f"rollout/opd/{key}"] = opd[key]

    # Reward statistics (mean / max / min / median)
    rewards = [float(s.reward) if not isinstance(s.reward, dict) else 0.0 for s in samples]
    if rewards:
        log_dict["rollout/reward/mean"] = float(np.mean(rewards))
        log_dict["rollout/reward/max"] = float(np.max(rewards))
        log_dict["rollout/reward/min"] = float(np.min(rewards))
        log_dict["rollout/reward/std"] = float(np.std(rewards))

    logger.info(f"rollout {rollout_id}: {log_dict}")
    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    log(args, log_dict, step_key="rollout/step")
    return True


def custom_eval_log(rollout_id, args, data, extra_metrics):
    """Custom eval metrics logger.

    For each eval dataset:
      eval/{name}/accuracy   - mean correctness (pass@1)
      eval/{name}/truncated_ratio

    Returns True to signal the framework that logging is handled here.
    """
    log_dict = dict(extra_metrics or {})

    for dataset_name, info in data.items():
        rewards = info["rewards"]       # flat list[float], 0.0 or 1.0
        truncated = info.get("truncated", [])
        prefix = f"eval/{dataset_name}"

        log_dict[f"{prefix}/accuracy"] = float(np.mean(rewards))

        if truncated:
            log_dict[f"{prefix}/truncated_ratio"] = float(np.mean(truncated))

    logger.info(f"eval {rollout_id}: {log_dict}")
    step = compute_rollout_step(args, rollout_id)
    # log_dict["rollout/step"] = step
    # log(args, log_dict, step_key="rollout/step")
    return True


