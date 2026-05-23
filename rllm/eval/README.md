# DeepResearch Eval

A simplified implementation of Vision DeepResearch.

## Files

| File | Purpose |
|---|---|
| `eval_runner.py` | Entry point: loads parquet, runs agents in parallel, judges answers, writes outputs |
| `deepresearch_agent.py` | `MultiTurnReactAgent` — multi-turn ReAct loop with tool calling |
| `tools.py` | Tool implementations (`search`, `visit`, `PythonInterpreter`, `crop_and_search`) |
| `unpack_parquet.py` | Extract images from a packed parquet → local files + rewritten parquet |
| `run_eval.sh` | Env-var wrapper around `eval_runner.py` (single file or whole directory) |
| `deploy/run_inference_server.sh` | vLLM launcher for the inference role (port 8001 by default) |
| `deploy/run_extract_judge_server.sh` | vLLM launcher for the shared extract + judge role (port 8002 by default) |
| `deploy/run_all_in_one_server.sh` | One-process vLLM launcher serving all roles (smoke-test convenience) |

---

## Step 1 — Install dependencies

```bash
pip install pandas pyarrow openai Pillow requests oss2 numpy json5
```


---

## Step 2 — Download and unpack the dataset

The packed parquets are hosted on HuggingFace:
**https://huggingface.co/datasets/Osilly/Vision-DeepResearch-Eval**
(files live under `data/` in the repo root).

```bash
pip install huggingface_hub

huggingface-cli download Osilly/Vision-DeepResearch-Eval \
    --repo-type dataset \
    --local-dir ./hf_data
```

Then unpack — this extracts embedded images to local files and rewrites
`images` to absolute local paths, dropping the `image_packed` column:

```bash
cd Vision-DeepResearch/rllm/eval

python unpack_parquet.py \
    --output_dir ./data/ready \
    --images_dir ./data/images \
    ../../../hf_data/data/*.parquet
```

`./data/ready/` will contain one inference-ready parquet per dataset;
`./data/images/` will hold the extracted image files referenced by them.

---

## Step 3 — Deploy the vLLM servers

The pipeline consumes three logical OpenAI-compatible roles:

| Role | Used by | Required env vars |
|---|---|---|
| **Inference** | `MultiTurnReactAgent` — produces tool calls and final answers | `TONGYI_BASE_URL`, `TONGYI_MODEL_NAME` |
| **Extract / summary** | `VisitTool` and `crop_and_search` — summarises raw webpage / visual-search HTML into structured JSON | `VLLM_EXTRACT_URL`, `EXTRACT_MODEL` |
| **Judge / reward** | `eval_runner.py` — LLM-as-judge for accuracy scoring | `JUDGE_BASE_URL`, `JUDGE_MODEL`, `JUDGE_API_KEY` |

The **extract (summary) and judge (reward) roles can share a single
server** — both are stateless OpenAI calls and a single VL-capable instruct
model (e.g. `Qwen3-VL-30B-A3B-Instruct`) covers both. The default deploy
below uses two processes total: one for inference, one shared by extract +
judge.

Ready-to-run launchers ship under `deploy/`:

| Script | Role | Default port |
|---|---|---|
| `deploy/run_inference_server.sh` | inference | 8001 |
| `deploy/run_extract_judge_server.sh` | extract + judge (shared) | 8002 |
| `deploy/run_all_in_one_server.sh` | all roles, single process | 8001 |

Each script accepts overrides via env vars: `MODEL_PATH`, `MODEL_NAME`,
`PORT`, `GPUS`, `TP_SIZE`, `GPU_MEM_UTIL`, `MAX_MODEL_LEN`. Typical
two-process deploy:



For a smoke test, run `deploy/run_all_in_one_server.sh` and point all three
URLs at the same port — slow at scale because traffic serialises through
one process.

---

## Step 4 — Configure environment variables

```bash
# --- Required: inference endpoint (port 8001) ---
export TONGYI_BASE_URL="http://localhost:8001/v1"
export TONGYI_MODEL_NAME="Qwen3-VL-30B-A3B-Instruct"
export TONGYI_API_KEY="EMPTY"             # vLLM does not check the key

# --- Extract / summary AND judge / reward (shared, port 8002) ---
export VLLM_EXTRACT_URL="http://localhost:8002"
export EXTRACT_MODEL="Qwen3-VL-30B-A3B-Instruct"

export JUDGE_BASE_URL="http://localhost:8002/v1"
export JUDGE_MODEL="Qwen3-VL-30B-A3B-Instruct"
export JUDGE_API_KEY="EMPTY"

# --- Required for crop_and_search tool ---
export ZHIPU_API_KEY="your_zhipu_key"        # Zhipu web/image search API
export OSS_ACCESS_KEY_ID="your_oss_ak"       # Aliyun OSS (stores cropped images)
export OSS_ACCESS_KEY_SECRET="your_oss_sk"
export OSS_BUCKET_NAME="your_oss_bucket"
export IMAGE_CROP_CACHE="/tmp/crop_cache"    # local temp dir for crops
```

---

## Step 5 — Run evaluation

### Via `run_eval.sh` (recommended)

`PARQUET` accepts either a single file **or a directory** — pointing it at
the directory will evaluate every `*.parquet` inside, writing one
sub-folder per dataset.

```bash
cd Vision-DeepResearch/rllm/eval

# Run all datasets at once
export PARQUET=./data/ready
export OUTPUT_DIR=./outputs/Qwen3-VL-30B-A3B-Instruct
bash run_eval.sh

# …or a single dataset
export PARQUET=./data/ready/bcvl_level1_199.parquet
bash run_eval.sh
```

Optional env vars for `run_eval.sh`:

| Variable | Default | Description |
|---|---|---|
| `PARALLEL_TASKS` | `10` | Concurrent tasks |
| `MAX_SAMPLES` | (all) | Cap number of samples |
| `TEMPERATURE` | `0.6` | Sampling temperature |
| `TOP_P` | `0.95` | Top-p |
| `MAX_TOKENS` | (unset) | Max completion tokens |
| `OUTPUT_DIR` | `./outputs` | Output root |
| `JUDGE_BASE_URL` | same as `TONGYI_BASE_URL` | Judge endpoint |
| `JUDGE_MODEL` | same as `TONGYI_MODEL_NAME` | Judge model |
| `JUDGE_API_KEY` | same as `TONGYI_API_KEY` | Judge API key |



## Datasets

| File | Samples |
|------|---------|
| bcvl_level1_199.parquet | 199 |
| bcvl_level2_200.parquet | 200 |
| fvqa_300.parquet | 300 |
| infoseek_300.parquet | 300 |
| livevqa_webwatcher_subset_300.parquet | 300 |
| mmsearch_vision_171.parquet | 171 |
| simplevqa_300.parquet | 300 |
| MMSearch-Plus_221.parquet | 221 |
| worldvqa_sampled_300.parquet | 300 |
| testmini500_v1.parquet   （VDR Bench） | 500 |

---

Thanks for their contributions.