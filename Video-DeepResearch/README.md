<p align="center">
  <h1 align="center">Video-DeepResearch</h1>
  <p align="center"><em>Towards the Next-Generation Multimodal Deepresearch Agent</em></p>
</p>

<p align="center">
    📑 <a href="https://arxiv.org/abs/2608.03979">Paper</a>
    &nbsp;·&nbsp;
    🌐 <a href="https://osilly.github.io/Vision-DeepResearch/">Project Page</a>
    &nbsp;·&nbsp;
    🤗 <a href="https://modelscope.cn/collections/CostaliyA/Video-DeepResearch">Models & Data</a>
    &nbsp;·&nbsp;
    🌐 <b>English</b> | <a href="README_zh.md">中文</a>
</p>

## 📢 News
- **[2026-08-08]** 🎉 We release the **Video-DeepResearch**-30B-A3B and **Video-DeepResearch**-35B-A3B ([Checkpoints](https://modelscope.cn/collections/CostaliyA/Video-DeepResearch)).
- **[2026-08-05]** 🎉 We release the **Video-DeepResearch** paper ([arXiv:2608.03979](https://arxiv.org/abs/2608.03979)).
- **[Coming soon]** 🚀 **VideoDR-Bench500** — an extended benchmark with **500 instances** covering harder, longer-horizon, and more diverse multi-hop tasks. Designed to push agents further along temporal grounding, cross-frame entity tracking, and multi-source knowledge synthesis. Stay tuned!

---

<p align="center">
  <img src="figs/paradigm.png" width="90%" alt="Video-DeepResearch pipeline overview">
</p>

Video-DeepResearch (Video-DR) extends multimodal deep-research agents from static images to **continuous video streams**, a setting that demands dense **spatiotemporal grounding** coupled with open-web exploration.

Preliminary experiments reveal two critical bottlenecks in current models: **(1) modality bias** — agents bypass visual tools in favor of textual search; **(2) parametric knowledge leakage** — models rely on internal memory rather than genuine tool-augmented execution.

To address these, Video-DR proposes:
- A **decoupled perception-exploration** data pipeline with **stage-wise tool unlocking** that forces the agent to complete cross-frame visual grounding before touching the web.
- A **two-stage training recipe**: SFT (7K trajectories + 7K text-only QA) → **GRPO** (2K moderate-difficulty, Pass@4 ∈ (0, 1), negative-advantage down-sampling at 20%).
- **VideoDR-Bench**: 200 human-AI collaboratively annotated multi-hop VQA instances, each provably requiring **both visual search and external-knowledge reasoning**. An extended **VideoDR-Bench500** with more challenging tasks is coming soon.

> **Video-DeepResearch-35B-A3B: 64.0% avg accuracy** — surpasses Claude-4.5-Sonnet (59.0%) by 5.0 points, and significantly outperforms GPT-5 (52.5%) and Gemini 2.5 Pro (57.5%). The 30B-A3B variant reaches 59.3%, on par with Claude-4.5-Sonnet.

---

## Repository Layout

```
Video-DeepResearch/
├── preprocess/   # 1. Data preprocessing: videos → keyframes
├── eval/         # 2. Evaluation: sglang / vllm / maas backends
├── sft/          # 3. Supervised fine-tuning: ms-swift + megatron
└── rl/           # 4. Reinforcement learning: slime + megatron + GRPO
```

The four sub-modules cover the full training and evaluation pipeline described in the paper (Fig. 1). Every sub-directory is self-contained — no global slime / ms-swift installation is required.

---

## 1. Data Preprocessing (preprocess/)

Extract visually distinctive keyframes from videos, written to `{output_dir}/{video_id}/frame_XXXX.XX.png`. CLIP-based cosine similarity filtering (via `--clip-model`) is preferred; if unavailable, it falls back to pixel-difference (faster but coarser). Multi-GPU parallelism is supported.

```bash
python3 preprocess/extract_keyframes.py \
    --video-dir  /path/to/videos \
    --output-dir /path/to/frames \
    --clip-model /path/to/clip-vit-large-patch14-336 \
    --max-frames 20 \
    --interval   1.0 \
    --threshold  0.80 \
    --num-gpus   8
```

**Key knobs**: `--interval` (sampling step, seconds) · `--threshold` (similarity cutoff — higher = more aggressive filtering) · `--max-frames` (per-video cap) · `--max-size` (long-edge resize).

Extracted frames feed directly into both `eval/` and RL rollout — they are the entry point for every Video-DR tool chain (input for `Select_Keyframe` + `Crop_Search` in paper §3).

---

## 2. Evaluation (eval/)

Three entry points corresponding to three deployment backends. All read `eval/config.yaml` (tool API keys, rollout working directory, reward server URL) and `eval/prompts/*.txt` (system prompts for tool / direct modes).

- **`run_eval_sglang.sh`** — local SGLang deployment; simplest.
- **`run_eval_vllm.sh`** — vLLM deployment; also supports `BACKEND=openai|claude` for closed-source proxies.
- **`run_eval_maas.sh`** — MaaS gateway (OpenAI-compatible), no local checkpoint required.

`eval/vdr_core/` is an embedded slime dependency slice (~17 files, zero external slime install). Key pieces: `rollout.py` with a local `GenerateState` shim (only exposes tokenizer/processor), `env.py` with the Gym-style multi-turn tool-interaction environment, and `slime_utils/` — a minimal slice of slime's native utilities.

**Prerequisites**: keyframes extracted, plus **three services running**:

1. **Inference server** — hosts the VLM under test (SGLang / vLLM). Passed as `inference_url` to the launcher.
2. **Judge server** — an OpenAI-compatible vLLM endpoint (paper uses Qwen3-VL-30B-A3B-Instruct) that scores each rollout's `<answer>`.
3. **Extract server** *(required)* — invoked by the `Visit` tool to summarize webpage contents into structured evidence. Without it, `Visit` degrades to returning raw HTML and agentic accuracy drops sharply.

The Extract server supports **two deployment backends**, selected via `extract_backend` in `eval/config.yaml`:

**Option A — vLLM (OpenAI-compatible):** shares one instance with the judge.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve \
    Qwen/Qwen3-VL-30B-A3B-Instruct \
    --host 0.0.0.0 --port 8001 \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.8 \
    --served-model-name "Qwen3-VL-30B-A3B-Instruct" \
    --max_model_len 160000 \
    --mm-processor-cache-gb 0 \
    --no-enable-prefix-caching
```

```yaml
# eval/config.yaml
extract_model:   Qwen3-VL-30B-A3B-Instruct
extract_backend: vllm       # POST /v1/chat/completions
extract_url:     http://<vllm-host>:8001/v1
```

**Option B — SGLang native `/generate`:** standalone SGLang server.

```bash
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-VL-30B-A3B-Instruct \
    --host 0.0.0.0 --port 13141 \
    --tp 8 --mem-fraction-static 0.8
```

```yaml
# eval/config.yaml
extract_model:   Qwen3-VL-30B-A3B-Instruct
extract_backend: sglang_generate   # POST /generate
extract_url:     http://<sglang-host>:13141/generate
```

The `extract_*` keys are auto-bridged to `EXTRACT_MODEL` / `EXTRACT_BACKEND` / `EXTRACT_URL` env vars by `env.build_env → _sync_tool_config_to_env`, then read by `visit_tool.py` (via `call_extract_model_async` in `vdr_core/tools/shared.py`) at runtime. The `eval/deploy/` directory ships helper scripts to spin up the inference and reward services.

```bash
# SGLang (open-source local deployment)
bash eval/run_eval_sglang.sh http://SGLANG_HOST:13141 http://JUDGE_HOST:8001 both Video-DR-35B-A3B

# vLLM (or openai/claude proxy via BACKEND=... switch)
bash eval/run_eval_vllm.sh   http://VLLM_HOST:8000/v1 http://JUDGE_HOST:8001 both Video-DR-35B-A3B

# MaaS (closed-source model, OpenAI-compatible gateway)
BACKEND=openai bash eval/run_eval_maas.sh "<API_KEY>" "https://<maas-host>/v1" http://JUDGE_HOST:8001 both qwen3.5-35b-a3b
```

**Output**: `eval/output/results/{model_name}/{mode}/` (mode ∈ {tool, direct, both}).

**Path overrides via env vars**: `CSV`, `FRAMES_DIR`, `OUTPUT_DIR`, `CONFIG`, `HF_CHECKPOINT`.

**Evaluation protocol** (paper §5): `Direct` lets the model answer from keyframes only (tool-free); `Agentic` exposes the full tool suite (`Select_Keyframe` / `Crop_Search` / `Search` / `Visit`) with multi-turn execution. Scoring runs against an independent vLLM judge server.

### Main Results (Paper Table 1, Agentic setting)

| Model | Video-DR | VideoDR-Bench Overall | **Avg** |
|:---|:---:|:---:|:---:|
| **Video-DeepResearch-35B-A3B** (Ours) | **72.4** | **71.2** | **64.0** |
| **Video-DeepResearch-30B-A3B** (Ours) | 68.0 | 67.5 | 59.3 |
| Claude-4.5-Sonnet | 66.2 | 69.5 | 59.0 |
| Gemini 2.5 Pro | 62.0 | 53.0 | 57.5 |
| GPT-5 | — | — | 52.5 |

<p align="center">
  <img src="figs/videohunt.png" width="95%" alt="VideoDR-Bench category distribution">
  <br><em>VideoDR-Bench spans six video domains — Knowledge, Entertainment, Daily Life, Game & Sports, News, Others — with every instance requiring joint visual grounding + multi-hop knowledge reasoning.</em>
</p>

---

## 3. Supervised Fine-Tuning (sft/)

Built on ms-swift's `megatron sft`, with Qwen3-VL-30B-A3B-Instruct (MoE, 256 experts / top-8) as the base. Cluster: 4 nodes × 8 × 80 GiB H800 (TP=4, EP=8, CP=2, PP=1; micro=1, global=64).

**Training data** (paper §4.3): 7K decoupled perception-exploration trajectories + 7K VDR text-only QA. Mixed training simultaneously reinforces visual-tool use and textual deep-research capability.

<p align="center">
  <img src="figs/pipeline.png" width="90%" alt="VideoHunter data pipeline">
  <br><em>Three-phase VideoHunter pipeline: (I) video filtering, (II) VQA synthesis with parametric-leakage filtering, (III) decoupled perception-exploration trajectory construction.</em>
</p>

```bash
# Single-node quick check
bash sft/run_video_dr_sft.sh

# Multi-node (run on each node; NODE_RANK is set by your scheduler)
WORLD_SIZE=4 RANK=$NODE_RANK bash sft/run_video_dr_sft.sh
```

**Env overrides**: `MODEL_PATH`, `DATASET_PATH` (space-separated multi-path), `SAVE_PATH`, `WANDB_KEY`, `NPROC_PER_NODE`.

**Data format** (per JSONL line): `messages` (multi-turn system/user/assistant, with `<image>` placeholders) + `images` (list of image paths aligned with placeholders).

`sft/ms-swift/` contains the ms-swift source (checkpoints / asset / docs / tests excluded). Install deps first: `pip install -r sft/ms-swift/requirements.txt`.

---

## 4. Reinforcement Learning (rl/)

GRPO training built on slime + megatron backend + SGLang rollout. Key hyperparameters (paper §4.3):

- **Reward**: sparse binary, `r=1` when the judge (Qwen3-VL-30B-A3B-Instruct) judges correct, else `r=0`.
- **Data**: 2K moderate-difficulty instances, Pass@4 strictly ∈ (0, 1).
- **Negative-advantage down-sampling**: format-violating / repetitive-loop trajectories contribute their negative gradient with only 20% probability (`--negative-advantage-keep-prob 0.2`).
- **Stability knobs**: `KL=0`, `ε_clip=0.2/0.28`, `--rollout-max-response-len 64000`, `--global-batch-size 512`.
- **Model parallelism**: TP=1, PP=2, EP=8, DP=8.

**Prerequisites**:
- Ray cluster is running (`SLIME_SCRIPT_EXTERNAL_RAY=1` + `RAY_JOB_ADDR`; or set to `0` to have the script start a local head)
- **Judge server** (vLLM, OpenAI-compatible) reachable at `JUDGE_IP:JUDGE_PORT/v1/models`
- **Extract server** *(required)* — same role as in eval: `visit_tool` invokes it during rollout to summarize webpage contents. Deploy exactly as in the eval section (SGLang `/generate` or vLLM `/v1/chat/completions`) and point `rl/examples/vision_deepresearch/config.yaml`'s `extract_backend` / `extract_url` / `extract_model` at it. In practice the same judge vLLM instance can double as extract.

```bash
export SLIME_SCRIPT_EXTERNAL_RAY=1
export SLIME_SCRIPT_NUM_NODES=2
export SLIME_SCRIPT_GPUS_PER_NODE=8
export SLIME_SCRIPT_RAY_JOB_ADDR="http://127.0.0.1:8265"
export SLIME_SCRIPT_JUDGE_IP="<judge-host>"
export SLIME_SCRIPT_JUDGE_PORT=8001
export SLIME_SCRIPT_TRAIN_DATA="/path/to/rollout.jsonl"

bash rl/run_grpo.sh
```

`rl/slime/` is a trimmed slime framework (~1 MB — utils / rollout / backends / ray only). `rl/scripts/models/` contains per-model megatron config scripts, `rl/train.py` is the slime entry, and `rl/examples/vision_deepresearch/` holds the vdr-side env / rollout / preprocess code.

---

## Known Issues

The code reorganization here (sub-directory split, import rewrites, slime shim extraction) introduces some divergences from the original upstream. Known caveats:

- **`eval/vdr_core/rollout.py`'s `GenerateState` is a local shim** — it only exposes tokenizer/processor and is a process-wide singleton (not keyed by hf_checkpoint). Fine for single-checkpoint eval; if you need to load multiple checkpoints in the same process or plug into full slime training, restore the original implementation.
- **`eval/vdr_core/env.py`'s `_judge` uses a soft slime.rollout.rm_hub dependency** (try/except ImportError): if full slime is installed you get a real score, otherwise it returns 0.0. Eval scoring actually flows through `run_eval.py`'s `DeepResearchReward` and is unaffected; RL training requires real slime.
- **`eval/vdr_core/env.py`'s system prompt path** changed from upstream's `Path(__file__).parent/"eval"/eval_system_prompt.txt` to `Path(__file__).parent.parent/"prompts"/eval_system_prompt.txt` (aligning with the new directory layout).
- **`eval/config.yaml` explicitly sets `rollout_interaction_env_path: vdr_core.env`** so `rollout.py` finds the local env module.
- **`eval/config.yaml` paths are relative (`./output/...`)** — if you launch scripts from a different directory, either `cd eval/` or convert them back to absolute paths.
- **Env vars consumed by eval** (`ZHIPU_API_KEY` / `OSS_ACCESS_KEY_ID` / `OSS_ACCESS_KEY_SECRET` / `IMAGE_CROP_CACHE` / `EXTRACT_URL`, etc.) are automatically bridged from config.yaml to `os.environ` inside `env.build_env → _sync_tool_config_to_env` — no manual export needed, provided the run goes through config.yaml.
- **All API keys / OSS credentials in config.yaml are placeholders** (e.g. `<YOUR_ZHIPU_API_KEY>`) — replace with valid values before running.
- **`sft/ms-swift/` contains source only, no checkpoints** — provide your own base model path. Install deps with `pip install -r sft/ms-swift/requirements.txt`.
- **`rl/slime/` is a minimal slime subset** (not full slime-2.4). If you need to hack internal slime logic, some modules may be missing — restore them from upstream.
- **`rl/run_grpo.sh`'s default `TRAIN_DATA_RAW` is a placeholder `/path/to/rollout.jsonl`** — must be overridden via `SLIME_SCRIPT_TRAIN_DATA`.
- **`preprocess/`'s `--clip-model` defaults to an empty string** — if left unset, extraction falls back to pixel-diff (faster, coarser).

For any reproduction issues, feel free to reach out: **fazii@mail.ustc.edu.cn**.

---

## Citation

```bibtex
@article{huang2026videodr,
  title   = {Video-DeepResearch: Towards the Next-Generation Multimodal Deepresearch Agent},
  author  = {Huang, Wenxuan and Zeng, Yu and Fang, Zhen and others},
  journal = {arXiv preprint arXiv:2608.03979},
  year    = {2026}
}
```
