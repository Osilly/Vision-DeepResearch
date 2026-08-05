import copy
import itertools
import json
import logging
import math
import os
import random
import re
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import ray
from tqdm import tqdm

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

from slime.utils.types import MultimodalTypes, Sample
from .timer import Timer

__all__ = ["Dataset"]

logger = logging.getLogger(__name__)

# =========================
# helpers
# =========================

def _batched(iterable, batch_size: int):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, batch_size))
        if not chunk:
            return
        yield chunk


def _estimate_num_rows(path: str):
    real_path, row_slice = _parse_generalized_path(path)

    if not os.path.exists(real_path):
        return None

    if real_path.endswith(".parquet") and pq is not None:
        total = pq.ParquetFile(real_path).metadata.num_rows
        if row_slice is None:
            return total
        start, stop, step = row_slice.indices(total)
        if step <= 0:
            return None
        return max(0, (stop - start + step - 1) // step)

    # jsonl 不强求统计总数，避免额外再扫一遍文件
    return None


def read_file(path):
    path, row_slice = _parse_generalized_path(path)
    reader = None

    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt dataset path '{path}' does not exist.")

    if path.endswith(".jsonl"):

        def jsonl_reader(p):
            with open(p, encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.warning("JSON decode error at line %s: %s", line_num, e)
                        continue

        reader = jsonl_reader(path)

    elif path.endswith(".parquet"):
        if pq is None:
            raise ImportError("pyarrow is required for parquet support")

        def parquet_reader(p):
            pf = pq.ParquetFile(p)
            # batch_size 可以按情况调大/调小
            for batch in pf.iter_batches(batch_size=2048):
                yield from batch.to_pylist()

        reader = parquet_reader(path)

    else:
        raise ValueError(f"Unsupported file format: {path}. Supported formats are .jsonl and .parquet.")

    if row_slice is not None:
        logger.info("read_file path=%s applying slice row_slice=%s", path, row_slice)
        reader = itertools.islice(reader, row_slice.start, row_slice.stop, row_slice.step)

    yield from reader


def _parse_generalized_path(s: str):
    if (m := re.match(r"^(?P<real_path>.*)@\[(?P<start>-?\d*):(?P<end>-?\d*)\]$", s)) is not None:
        path = m.group("real_path")
        start = int(x) if (x := m.group("start")) != "" else None
        end = int(x) if (x := m.group("end")) != "" else None
        return path, slice(start, end)

    return s, None


def _build_messages(data: dict, prompt_key: str, as_conversation: bool, multimodal_keys: dict = None):
    prompt = data.get(prompt_key)

    if prompt is None:
        available_keys = sorted(data.keys())
        raise ValueError(
            f"Dataset record is missing prompt key '{prompt_key}' or its value is None. "
            f"Available keys: {available_keys}"
        )

    if isinstance(prompt, str):
        if not as_conversation:
            return prompt
        prompt = [{"role": "user", "content": prompt}]

    if not isinstance(prompt, list):
        raise ValueError(
            f"Prompt field '{prompt_key}' must be a string or a list of messages, got {type(prompt)} instead"
        )

    if multimodal_keys:
        multimodals = {}
        for type_name, data_key in multimodal_keys.items():
            mt = MultimodalTypes.get(type_name)
            if mt:
                multimodal_data = data.get(data_key)
                if multimodal_data is not None:
                    multimodals[mt.placeholder] = (mt, list(multimodal_data))

        if multimodals:
            pattern = "(" + "|".join(re.escape(p) for p in multimodals.keys()) + ")"

            for message in prompt:
                if isinstance(message["content"], str):
                    content_list = []
                    for segment in re.split(pattern, message["content"]):
                        if not segment:
                            continue
                        if segment in multimodals:
                            mt, content = multimodals[segment]
                            assert len(content) > 0, (
                                f"Not enough {mt.name} data: more '{mt.placeholder}' placeholders in prompt "
                                f"than {mt.name}s provided in data"
                            )
                            content_list.append({"type": mt.name, mt.name: content.pop(0)})
                        else:
                            content_list.append({"type": "text", "text": segment})
                    message["content"] = content_list

                elif isinstance(message["content"], list):
                    continue
                else:
                    raise ValueError(
                        f"Unsupported content type: {type(message['content'])}, expected str or list of dicts"
                    )

            for placeholder, (mt, remaining) in multimodals.items():
                assert len(remaining) == 0, (
                    f"Multimodal data count mismatch: {len(remaining)} more {mt.name}(s)"
                    f" than '{placeholder}' placeholders in prompt"
                )

    return prompt


def _build_multimodal_inputs(prompt, processor):
    from slime.utils.processing_utils import process_vision_info

    assert isinstance(prompt, list), f"prompt must be a list when processor is not None, got {type(prompt)} instead"
    return process_vision_info(prompt, processor)


def _load_process_local_tokenizer_and_processor(tokenizer, processor):
    from slime.utils.processing_utils import load_processor, load_tokenizer

    if isinstance(tokenizer, str):
        loaded_tokenizer = load_tokenizer(tokenizer, trust_remote_code=True)
    else:
        loaded_tokenizer = tokenizer

    if isinstance(processor, str):
        loaded_processor = load_processor(processor, trust_remote_code=True)
    else:
        loaded_processor = processor

    return loaded_tokenizer, loaded_processor


def _process_single_record(
    data: dict,
    *,
    tokenizer,
    processor,
    prompt_key: str,
    multimodal_keys: dict | None,
    label_key: str | None,
    tool_key: str | None,
    metadata_key: str,
    apply_chat_template: bool,
    apply_chat_template_kwargs: dict | None,
):
    as_conversation = apply_chat_template or (multimodal_keys is not None)
    prompt = _build_messages(data, prompt_key, as_conversation, multimodal_keys)
    raw_prompt = copy.deepcopy(prompt)

    metadata = data.get(metadata_key) or {}
    tools = None
    if tool_key is not None and tool_key in data:
        tools = data[tool_key]
        if isinstance(tools, str):
            tools = json.loads(tools)
        elif isinstance(tools, np.ndarray):
            tools = tools.tolist()
        assert isinstance(tools, list), f"tools must be a list, got {type(tools)} instead"
        metadata["tools"] = tools

    if apply_chat_template:
        output_prompt = tokenizer.apply_chat_template(
            prompt,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            **(apply_chat_template_kwargs or {}),
        )
    else:
        output_prompt = prompt

    multimodal_inputs = _build_multimodal_inputs(prompt, processor) if processor else None
    metadata["raw_prompt"] = raw_prompt

    return Sample(
        prompt=output_prompt,
        label=data[label_key] if label_key is not None else None,
        metadata=metadata,
        multimodal_inputs=multimodal_inputs,
    )


def _get_prompt_length(sample: Sample, tokenizer, processor):
    # 保持和你原本语义一致：只有 prompt 是字符串时才做 max_length 过滤
    if not isinstance(sample.prompt, str):
        return None

    is_mm = (
        processor is not None
        and sample.multimodal_inputs is not None
        and any(v is not None for v in sample.multimodal_inputs.values())
    )

    if is_mm:
        encoded = processor(text=sample.prompt, **sample.multimodal_inputs)
        return len(encoded["input_ids"][0])

    input_ids = tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
    return len(input_ids)


# =========================
# process-pool globals
# =========================

_WORKER_TOKENIZER = None
_WORKER_PROCESSOR = None
_WORKER_CFG = None


def _init_dataset_worker(tokenizer_ref, processor_ref, worker_cfg):
    global _WORKER_TOKENIZER, _WORKER_PROCESSOR, _WORKER_CFG
    _WORKER_TOKENIZER, _WORKER_PROCESSOR = _load_process_local_tokenizer_and_processor(
        tokenizer_ref, processor_ref
    )
    _WORKER_CFG = worker_cfg


def _process_record_batch_mp(data_batch: list[dict]):
    out = []
    dropped = 0

    for data in data_batch:
        sample = _process_single_record(
            data,
            tokenizer=_WORKER_TOKENIZER,
            processor=_WORKER_PROCESSOR,
            prompt_key=_WORKER_CFG["prompt_key"],
            multimodal_keys=_WORKER_CFG["multimodal_keys"],
            label_key=_WORKER_CFG["label_key"],
            tool_key=_WORKER_CFG["tool_key"],
            metadata_key=_WORKER_CFG["metadata_key"],
            apply_chat_template=_WORKER_CFG["apply_chat_template"],
            apply_chat_template_kwargs=_WORKER_CFG["apply_chat_template_kwargs"],
        )

        max_length = _WORKER_CFG["max_length"]
        if max_length is not None:
            prompt_len = _get_prompt_length(sample, _WORKER_TOKENIZER, _WORKER_PROCESSOR)
            if prompt_len is not None:
                if prompt_len > max_length:
                    dropped += 1
                    continue
                sample.metadata["prompt_length"] = prompt_len

        out.append(sample)

    return len(data_batch), dropped, out


def _process_records_in_process_pool(
    path,
    process_config,
    max_workers,
    progress_desc,
    task_batch_size=128,
    mp_start_method="fork",
):
    total_rows = _estimate_num_rows(path)

    # Linux 上如果你确认环境稳定，可以改成 fork，通常更快
    # 但默认保守一点，仍用 spawn
    ctx = mp.get_context(mp_start_method)

    all_samples = []
    dropped_total = 0

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=_init_dataset_worker,
        initargs=(
            process_config["tokenizer"],
            process_config["processor"],
            process_config,
        ),
    ) as executor:
        batch_iter = _batched(read_file(path), task_batch_size)
        mapped = executor.map(_process_record_batch_mp, batch_iter, chunksize=1)

        with tqdm(total=total_rows, desc=progress_desc, unit="samples") as pbar:
            for input_count, dropped, samples in mapped:
                all_samples.extend(samples)
                dropped_total += dropped
                pbar.update(input_count)

    if process_config["max_length"] is not None:
        logger.info(
            "Filtered %s samples longer than max_length=%s during parallel loading.",
            dropped_total,
            process_config["max_length"],
        )

    return all_samples


class Dataset:
    def __init__(
        self,
        path,
        tokenizer,
        processor,
        max_length,
        *,
        prompt_key="text",
        multimodal_keys=None,
        label_key=None,
        tool_key=None,
        metadata_key="metadata",
        seed=42,
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
        multimodal_load_workers=0,
        multimodal_task_batch_size=128,
        mp_start_method="spawn",
    ):
        progress_desc = f"Loading dataset {os.path.basename(_parse_generalized_path(path)[0])}"

        # 只有在真正需要多进程时才走 pool
        if processor is not None and multimodal_load_workers and multimodal_load_workers > 1:
            tokenizer_ref = getattr(tokenizer, "name_or_path", tokenizer)
            processor_ref = getattr(processor, "name_or_path", processor)

            process_config = {
                "tokenizer": tokenizer_ref,
                "processor": processor_ref,
                "prompt_key": prompt_key,
                "multimodal_keys": multimodal_keys,
                "label_key": label_key,
                "tool_key": tool_key,
                "metadata_key": metadata_key,
                "apply_chat_template": apply_chat_template,
                "apply_chat_template_kwargs": apply_chat_template_kwargs,
                "max_length": max_length,
            }

            logger.info(
                "Loading multimodal dataset with %s worker processes from %s "
                "(task_batch_size=%s, start_method=%s)",
                multimodal_load_workers,
                path,
                multimodal_task_batch_size,
                mp_start_method,
            )

            self.origin_samples = _process_records_in_process_pool(
                path=path,
                process_config=process_config,
                max_workers=multimodal_load_workers,
                progress_desc=progress_desc,
                task_batch_size=multimodal_task_batch_size,
                mp_start_method=mp_start_method,
            )
        else:
            total_rows = _estimate_num_rows(path)
            origin_samples = []
            dropped = 0

            for data in tqdm(read_file(path), total=total_rows, desc=progress_desc, unit="samples"):
                sample = _process_single_record(
                    data,
                    tokenizer=tokenizer,
                    processor=processor,
                    prompt_key=prompt_key,
                    multimodal_keys=multimodal_keys,
                    label_key=label_key,
                    tool_key=tool_key,
                    metadata_key=metadata_key,
                    apply_chat_template=apply_chat_template,
                    apply_chat_template_kwargs=apply_chat_template_kwargs,
                )

                if max_length is not None:
                    prompt_len = _get_prompt_length(sample, tokenizer, processor)
                    if prompt_len is not None:
                        if prompt_len > max_length:
                            dropped += 1
                            continue
                        sample.metadata["prompt_length"] = prompt_len

                origin_samples.append(sample)

            if max_length is not None:
                logger.info(
                    "Filtered %s samples longer than max_length=%s during serial loading.",
                    dropped,
                    max_length,
                )

            self.origin_samples = origin_samples

        self.epoch_id = -1
        self.seed = seed
        self.samples = self.origin_samples

    def shuffle(self, new_epoch_id):
        if self.epoch_id == new_epoch_id:
            return

        random.seed(self.seed + new_epoch_id)
        permutation = list(range(len(self.samples)))
        random.shuffle(permutation)
        self.samples = [self.origin_samples[i] for i in permutation]
        self.epoch_id = new_epoch_id

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


def get_minimum_num_micro_batch_size(total_lengths, max_tokens_per_gpu):
    batches = []
    for length in total_lengths:
        for i in range(len(batches)):
            if batches[i] + length <= max_tokens_per_gpu:
                batches[i] += length
                break
        else:
            batches.append(length)
    return len(batches)


def process_rollout_data(args, rollout_data_ref, dp_rank, dp_size):
    assert len(rollout_data_ref) == dp_size
    rollout_data = ray.get(rollout_data_ref[dp_rank].inner)

    partition = rollout_data.pop("partition")
    total_lengths = rollout_data["total_lengths"]

    Timer().seq_lens = total_lengths
    rollout_data["total_lengths"] = [total_lengths[i] for i in partition]

    return rollout_data