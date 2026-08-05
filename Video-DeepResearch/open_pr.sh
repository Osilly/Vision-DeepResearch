#!/usr/bin/env bash
# open_pr.sh — Stage the Video-DeepResearch/ addition + outer README.md news
# line onto a fresh feature branch, push to origin, and open a PR against
# main. Excludes this script itself from the commit.
#
# Prerequisites:
#   - You have push access to origin (or a fork you've added as a remote).
#   - `git` is installed.
#   - `gh` CLI is optional; when missing, the script prints a browser URL
#     you can click to open the PR from GitHub's compare page.
#
# Usage:
#   bash Video-DeepResearch/open_pr.sh
#
#   # Override the auto-generated branch name / PR title
#   BRANCH=feat/video-dr TITLE="Add Video-DR pipeline" bash Video-DeepResearch/open_pr.sh
#
#   # Push to a different remote (e.g. your own fork)
#   REMOTE=fork bash Video-DeepResearch/open_pr.sh

set -euo pipefail

# ---- locate the outer repo (walk up from this script) --------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_FILE="${BASH_SOURCE[0]}"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "${REPO_ROOT}" ]; then
    echo "[error] not inside a git repo: ${SCRIPT_DIR}" >&2
    exit 1
fi
cd "${REPO_ROOT}"

# Relative path (from repo root) to this script — excluded from the commit.
SCRIPT_REL="$(realpath --relative-to="${REPO_ROOT}" "${SCRIPT_FILE}")"

# ---- config --------------------------------------------------------------
REMOTE="${REMOTE:-origin}"
BASE_BRANCH="${BASE_BRANCH:-main}"
BRANCH="${BRANCH:-feat/video-deepresearch-$(date +%Y%m%d-%H%M%S)}"
TITLE="${TITLE:-Add Video-DeepResearch subrepo (preprocess / eval / sft / rl)}"

# Files we intend to stage — anything else (including this script) is left alone.
PATHS=(
    "Video-DeepResearch"
    "README.md"
)

# ---- pre-flight ----------------------------------------------------------
echo "============================================================"
echo " repo   : ${REPO_ROOT}"
echo " remote : ${REMOTE}  ($(git remote get-url "${REMOTE}" 2>/dev/null || echo '<not set>'))"
echo " base   : ${BASE_BRANCH}"
echo " branch : ${BRANCH}"
echo " title  : ${TITLE}"
echo " paths  : ${PATHS[*]}"
echo " exclude: ${SCRIPT_REL}  (this script itself)"
echo "============================================================"

if ! git remote get-url "${REMOTE}" > /dev/null 2>&1; then
    echo "[error] remote '${REMOTE}' not configured. Add one with:" >&2
    echo "        git remote add ${REMOTE} <url>" >&2
    exit 1
fi

for p in "${PATHS[@]}"; do
    if [ ! -e "${p}" ]; then
        echo "[error] path missing: ${p}" >&2
        exit 1
    fi
done

# ---- checkout feature branch --------------------------------------------
CURRENT_BRANCH="$(git branch --show-current)"
echo ""
echo "[step 1/5] Creating branch ${BRANCH} (from ${CURRENT_BRANCH})"
if git show-ref --verify --quiet "refs/heads/${BRANCH}"; then
    echo "  branch already exists — checking it out"
    git checkout "${BRANCH}"
else
    git checkout -b "${BRANCH}"
fi

# ---- stage + commit (excluding this script itself) ----------------------
echo ""
echo "[step 2/5] Staging paths (excluding ${SCRIPT_REL})"
git add -- "${PATHS[@]}"
# Unstage this script if it snuck in under Video-DeepResearch/
if git ls-files --cached --error-unmatch "${SCRIPT_REL}" > /dev/null 2>&1 \
   || git diff --cached --name-only | grep -Fxq "${SCRIPT_REL}"; then
    git restore --staged -- "${SCRIPT_REL}" 2>/dev/null || git reset HEAD -- "${SCRIPT_REL}"
    echo "  unstaged ${SCRIPT_REL}"
fi

echo ""
echo "[step 3/5] Committing"
if git diff --cached --quiet; then
    echo "  no staged changes — skipping commit"
else
    git commit -m "$(cat <<'EOF'
Add Video-DeepResearch subrepo

Adds Video-DeepResearch/, a self-contained code layout covering the full
Video-DR pipeline from the arXiv:2608.03979 paper:

  * preprocess/  keyframe extraction (CLIP-based, multi-GPU).
  * eval/        vdr_core (embedded slime dependency slice, no external
                 slime install) + three launch backends (sglang / vllm /
                 maas), config-driven Extract server support (sglang or
                 vllm) required by the Visit tool.
  * sft/         ms-swift megatron SFT launcher + vendored ms-swift
                 source (checkpoints/asset/docs/tests excluded).
  * rl/          slime + megatron + sglang GRPO launcher, minimal slime
                 subset (~1 MB).

Also updates the top-level README news section with the 2026-08-05
Video-DR paper release entry.

All secrets / internal paths in Video-DeepResearch/ are placeholders
(<YOUR_ZHIPU_API_KEY>, /path/to/..., etc.); no proprietary content
is committed.
EOF
)"
fi

# ---- push ---------------------------------------------------------------
echo ""
echo "[step 4/5] Pushing to ${REMOTE}/${BRANCH}"
git push -u "${REMOTE}" "${BRANCH}"

# ---- open PR ------------------------------------------------------------
echo ""
echo "[step 5/5] Opening PR against ${BASE_BRANCH}"
REMOTE_URL="$(git remote get-url "${REMOTE}")"
# Normalize git@github.com:owner/repo(.git) → owner/repo
SLUG="$(echo "${REMOTE_URL}" \
    | sed -E 's|^git@github\.com:||; s|^https?://github\.com/||; s|\.git$||')"

if command -v gh > /dev/null 2>&1; then
    gh pr create \
        --base "${BASE_BRANCH}" \
        --head "${BRANCH}" \
        --title "${TITLE}" \
        --body "$(cat <<'EOF'
## Summary

Adds `Video-DeepResearch/` — a self-contained subrepo for the
[Video-DeepResearch (arXiv:2608.03979)](https://arxiv.org/abs/2608.03979) paper.

Four modules mirroring the paper's pipeline (Fig. 1):
- **preprocess/** — CLIP-based multi-GPU keyframe extraction.
- **eval/** — three launch backends (sglang / vllm / maas); embedded
  slime dependency slice (`vdr_core/`) so no external slime install is
  needed; Extract server (sglang `/generate` or vllm `/v1/chat/completions`)
  is documented as required for the `Visit` tool.
- **sft/** — ms-swift megatron SFT launcher + vendored ms-swift source
  (checkpoints / asset / docs / tests excluded).
- **rl/** — slime + megatron + sglang GRPO launcher; minimal slime
  subset (~1 MB) with `train.py` / `scripts/models/` / `examples/vision_deepresearch/`.

Also updates the top-level `README.md` news timeline with the
2026-08-05 paper release entry.

## Test plan

- [ ] `python3 -c 'import sys; sys.path.insert(0, "Video-DeepResearch/eval"); from vdr_core.rollout import generate'` succeeds
- [ ] `bash -n Video-DeepResearch/eval/run_eval_{sglang,vllm,maas}.sh` all pass
- [ ] `bash -n Video-DeepResearch/sft/run_video_dr_sft.sh` passes
- [ ] `bash -n Video-DeepResearch/rl/run_grpo.sh` passes
- [ ] `grep -rlE '/mnt/tidal-alsh01|ZHIPU_API_KEY.*=[^<]' Video-DeepResearch/` returns empty (no secrets or internal paths)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
else
    COMPARE_URL="https://github.com/${SLUG}/compare/${BASE_BRANCH}...${BRANCH}?expand=1"
    echo ""
    echo "  gh CLI not found — install it (https://cli.github.com) to auto-open"
    echo "  the PR, or click the URL below to open the compare page in your browser:"
    echo ""
    echo "  ${COMPARE_URL}"
fi

echo ""
echo "Done. Branch ${BRANCH} pushed to ${REMOTE}; open_pr.sh was not committed."
