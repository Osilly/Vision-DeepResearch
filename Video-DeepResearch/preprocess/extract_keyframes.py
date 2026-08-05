#!/usr/bin/env python3
"""
Keyframe extractor for evaluation videos.

Uses CLIP cosine similarity to select visually distinct frames.
Falls back to per-pixel frame difference when CLIP is unavailable.

Key design choices for speed:
  - Seek-based frame access (cap.set) instead of sequential grab loop
  - Batched CLIP inference (--batch-size frames per forward pass)
  - Two-pass: batch-encode all candidates, then filter by similarity
  - Multi-GPU: one subprocess per GPU via multiprocessing.spawn

Output layout:
  {output_dir}/{video_id}/frame_XXXX.XX.png

Usage:
  python3 extract_keyframes.py \
    --video-dir /path/to/videos \
    --output-dir /path/to/frames \
    [--clip-model /path/to/clip-model] \
    [--max-frames 20] \
    [--interval 1.0] \
    [--threshold 0.80] \
    [--max-size 1024] \
    [--batch-size 64] \
    [--num-gpus 8] \
    [--device cuda]
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)

VIDEO_EXTS = {'.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', '.webm'}


# ---------------------------------------------------------------------------
# Frame utilities
# ---------------------------------------------------------------------------

def is_pure_frame(frame: np.ndarray, threshold: float = 1.0) -> bool:
    return np.std(frame) < threshold


def get_resized_wh(width: int, height: int, max_size: int):
    if width > max_size or height > max_size:
        scale = max_size / max(width, height)
        return int(width * scale), int(height * scale)
    return width, height


def _pixel_similarity(ref_frame: np.ndarray, curr_frame: np.ndarray) -> float:
    ref_g = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    cur_g = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return 1.0 - float(np.mean(np.abs(ref_g - cur_g))) / 255.0


def _limit_frames(idxs: list[int], max_save: int) -> list[int]:
    if len(idxs) <= max_save:
        return idxs
    first, last = idxs[0], idxs[-1]
    mid_count = max_save - 2
    if mid_count <= 0:
        return [first, last] if max_save >= 2 else [first]
    mid = sorted(random.sample(idxs[1:-1], min(len(idxs) - 2, mid_count)))
    return [first] + mid + [last]


def _seek_read(cap: cv2.VideoCapture, idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    return frame if ret else None


# ---------------------------------------------------------------------------
# CLIP-based extractor
# ---------------------------------------------------------------------------

class KeyframeExtractor:
    def __init__(self, clip_model_path: str | None = None, device: str = "cuda",
                 batch_size: int = 64):
        self._clip = None
        self._batch_size = batch_size
        if clip_model_path and os.path.exists(clip_model_path):
            try:
                import torch
                from transformers import CLIPFeatureExtractor, CLIPVisionModel

                dev = torch.device(device if torch.cuda.is_available() else "cpu")
                fe = CLIPFeatureExtractor.from_pretrained(clip_model_path)
                vm = CLIPVisionModel.from_pretrained(clip_model_path).to(dev).eval()
                self._clip = (torch, dev, fe, vm)
                log.info("CLIP loaded from %s on %s", clip_model_path, dev)
            except Exception as exc:
                log.warning("CLIP load failed (%s) — pixel-diff fallback", exc)

    def _encode_batch(self, frames: list[np.ndarray]):
        """BGR frames → CLIP features tensor (N, D)."""
        torch, dev, fe, vm = self._clip
        rgbs = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        inputs = fe(images=rgbs, return_tensors="pt").to(dev)
        with torch.no_grad():
            return vm(**inputs).last_hidden_state[:, 0, :]

    # ------------------------------------------------------------------

    def process_video(
        self,
        video_path: str,
        output_dir: str,
        *,
        interval: float = 1.0,
        threshold: float = 0.80,
        max_frames: int = 20,
        max_size: int = 1024,
        max_duration: float = 1800.0,
    ) -> dict:
        video_name = os.path.basename(video_path)

        if os.path.exists(output_dir) and any(
            f.endswith('.png') for f in os.listdir(output_dir)
        ):
            return {"status": "skip", "video_name": video_name}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"status": "failed", "reason": "open_fail", "video_name": video_name}

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0:
            cap.release()
            return {"status": "failed", "reason": "fps_0", "video_name": video_name}

        frame_step = max(1, math.ceil(fps * interval))
        max_idx    = min(total_frames - 1, int(max_duration * fps))
        new_w, new_h = get_resized_wh(width, height, max_size)

        # All candidate frame indices — seek directly, no grab loop
        candidate_idxs = list(range(0, max_idx + 1, frame_step))

        if self._clip:
            selected_idxs = self._clip_select(cap, candidate_idxs, threshold)
        else:
            selected_idxs = self._pixel_select(cap, candidate_idxs, threshold)

        cap.release()

        if not selected_idxs:
            return {"status": "failed", "reason": "no_valid_frames", "video_name": video_name}

        final_idxs = _limit_frames(selected_idxs, max_frames)
        os.makedirs(output_dir, exist_ok=True)

        cap2 = cv2.VideoCapture(video_path)
        saved = 0
        for idx in final_idxs:
            frame = _seek_read(cap2, idx)
            if frame is not None:
                if (new_w, new_h) != (width, height):
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                time_mark = f"{idx / fps:07.2f}"
                out_path = os.path.join(output_dir, f"frame_{time_mark}.png")
                cv2.imwrite(out_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                saved += 1
        cap2.release()

        return {
            "status": "success",
            "video_name": video_name,
            "output_dir": output_dir,
            "saved_count": saved,
        }

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------

    def _clip_select(self, cap: cv2.VideoCapture, candidate_idxs: list[int],
                     threshold: float) -> list[int]:
        import torch
        import torch.nn.functional as F

        # Pass 1: read candidates in seek order, batch-encode
        valid_idxs: list[int] = []
        feat_chunks = []
        batch_frames: list[np.ndarray] = []

        def flush():
            if batch_frames:
                feat_chunks.append(self._encode_batch(batch_frames))
                batch_frames.clear()

        for idx in candidate_idxs:
            frame = _seek_read(cap, idx)
            if frame is None:
                continue
            if not valid_idxs and is_pure_frame(frame):
                continue
            valid_idxs.append(idx)
            batch_frames.append(frame)
            if len(batch_frames) >= self._batch_size:
                flush()
        flush()

        if not valid_idxs:
            return []

        all_feats = torch.cat(feat_chunks, dim=0)  # (N, D)

        # Pass 2: similarity filter on pre-computed tensors
        selected = [valid_idxs[0]]
        ref = all_feats[0:1]
        for k in range(1, len(valid_idxs)):
            feat = all_feats[k:k+1]
            if F.cosine_similarity(ref, feat, dim=1).item() < threshold:
                selected.append(valid_idxs[k])
                ref = feat

        return selected

    def _pixel_select(self, cap: cv2.VideoCapture, candidate_idxs: list[int],
                      threshold: float) -> list[int]:
        selected: list[int] = []
        ref_frame = None
        for idx in candidate_idxs:
            frame = _seek_read(cap, idx)
            if frame is None:
                continue
            if not selected:
                if is_pure_frame(frame):
                    continue
                ref_frame = frame
                selected.append(idx)
            else:
                if _pixel_similarity(ref_frame, frame) < threshold:
                    selected.append(idx)
                    ref_frame = frame
        return selected


# ---------------------------------------------------------------------------
# Multi-GPU worker (spawned subprocess)
# ---------------------------------------------------------------------------

def _gpu_worker(gpu_idx: int, tasks: list[tuple[str, str]], cfg: dict) -> list[dict]:
    logging.basicConfig(level=logging.INFO, format=f"[GPU{gpu_idx}] %(asctime)s %(message)s")
    extractor = KeyframeExtractor(
        clip_model_path=cfg["clip_model"],
        device=f"cuda:{gpu_idx}",
        batch_size=cfg.get("batch_size", 64),
    )
    results = []
    for video_path, output_dir in tasks:
        res = extractor.process_video(
            video_path, output_dir,
            interval=cfg["interval"],
            threshold=cfg["threshold"],
            max_frames=cfg["max_frames"],
            max_size=cfg["max_size"],
            max_duration=cfg["max_duration"],
        )
        res["vid_id"] = Path(video_path).stem
        results.append(res)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Extract keyframes from evaluation videos")
    p.add_argument("--video-dir",  required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--clip-model",
        default="",
    )
    p.add_argument("--max-frames",   type=int,   default=20)
    p.add_argument("--interval",     type=float, default=1.0)
    p.add_argument("--threshold",    type=float, default=0.80)
    p.add_argument("--max-size",     type=int,   default=1024)
    p.add_argument("--max-duration", type=float, default=1800.0)
    p.add_argument("--batch-size",   type=int,   default=64,
                   help="CLIP batch size per forward pass")
    p.add_argument("--num-gpus",     type=int,   default=1,
                   help="Number of GPUs to use in parallel")
    p.add_argument("--device",       default="cuda",
                   help="Device for single-GPU path (e.g. cuda, cuda:2, cpu)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    video_dir  = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    videos     = sorted(v for v in video_dir.iterdir() if v.suffix.lower() in VIDEO_EXTS)
    log.info("Found %d videos in %s", len(videos), video_dir)

    tasks = [(str(vp), str(output_dir / vp.stem)) for vp in videos]

    cfg = dict(
        clip_model=args.clip_model,
        interval=args.interval,
        threshold=args.threshold,
        max_frames=args.max_frames,
        max_size=args.max_size,
        max_duration=args.max_duration,
        batch_size=args.batch_size,
    )

    all_results: list[dict] = []

    if args.num_gpus <= 1:
        extractor = KeyframeExtractor(
            clip_model_path=args.clip_model,
            device=args.device,
            batch_size=args.batch_size,
        )
        for vp, out in tqdm(tasks, desc="Extracting keyframes"):
            res = extractor.process_video(
                vp, out,
                interval=args.interval,
                threshold=args.threshold,
                max_frames=args.max_frames,
                max_size=args.max_size,
                max_duration=args.max_duration,
            )
            res["vid_id"] = Path(vp).stem
            all_results.append(res)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")

        chunks: list[list] = [[] for _ in range(args.num_gpus)]
        for i, t in enumerate(tasks):
            chunks[i % args.num_gpus].append(t)

        log.info("Distributing %d videos across %d GPUs (%s each)",
                 len(tasks), args.num_gpus,
                 ", ".join(str(len(c)) for c in chunks))

        with ctx.Pool(processes=args.num_gpus) as pool:
            futures = [
                pool.apply_async(_gpu_worker, (gpu_idx, chunk, cfg))
                for gpu_idx, chunk in enumerate(chunks)
                if chunk
            ]
            for fut in tqdm(futures, desc="GPU workers", unit="worker"):
                all_results.extend(fut.get())

    counts: dict[str, int] = {}
    for res in all_results:
        status = res.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
        if status == "success":
            log.info("  %-6s  %d frames -> %s",
                     res.get("vid_id", "?"), res["saved_count"], res.get("output_dir", ""))
        elif status == "failed":
            log.warning("  %-6s  FAILED: %s", res.get("vid_id", "?"), res.get("reason"))

    log.info("Done. success=%d  skip=%d  failed=%d",
             counts.get("success", 0), counts.get("skip", 0), counts.get("failed", 0))


if __name__ == "__main__":
    main()
