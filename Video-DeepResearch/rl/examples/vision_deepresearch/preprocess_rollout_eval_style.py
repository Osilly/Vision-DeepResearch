#!/usr/bin/env python3
"""Preprocess rollout.jsonl into eval-aligned messages format for RL rollout.

Mirrors examples/vision_deepresearch/eval/run_eval.py: each row becomes a
[system, user] message pair where
  - system  = eval_system_prompt.txt with {tools} (JSON-serialized tool schemas
              from tools/registry.py, filtered by --supported-tools) and {date}
              substituted in;
  - user    = "The following are frames sampled from a video clip." +
              per-frame "image_id: image_N\\n" labels + {"type":"image"} entries
              + "\\n\\nQuestion: <q>".

Training reads the output file with --input-key messages (instead of question).
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

from examples.vision_deepresearch.tools.registry import get_tools


def build_system_content(template: str, supported_tool_names: list[str]) -> str:
    tool_map = {t["function"]["name"]: t for t in get_tools()}
    tools_json = "\n".join(
        json.dumps(tool_map[name], ensure_ascii=False)
        for name in supported_tool_names
        if name in tool_map
    )
    date_str = datetime.date.today().isoformat()
    return template.replace("{tools}", tools_json).replace("{date}", date_str)


def build_user_content(image_paths: list[str], question: str) -> list[dict]:
    if not image_paths:
        return [{"type": "text", "text": question}]
    parts: list[dict] = [
        {"type": "text", "text": "The following are frames sampled from a video clip.\n"}
    ]
    for i, path in enumerate(image_paths):
        parts.append({"type": "text", "text": f"image_id: image_{i}\n"})
        parts.append({"type": "image", "image": path})
    parts.append({"type": "text", "text": f"\n\nQuestion: {question}"})
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="rollout.jsonl with {question, images, label, ...}")
    ap.add_argument("--output", required=True, help="output jsonl with {messages, label, images, ...}")
    ap.add_argument(
        "--system-prompt-file",
        default=str(Path(__file__).parent / "eval" / "eval_system_prompt.txt"),
    )
    ap.add_argument(
        "--supported-tools",
        default="search,visit,select_crop_search",
        help="Comma-separated tool names; matches eval/config.yaml default.",
    )
    args = ap.parse_args()

    template = Path(args.system_prompt_file).read_text(encoding="utf-8")
    supported = [t.strip() for t in args.supported_tools.split(",") if t.strip()]
    system_content = build_system_content(template, supported)

    n_in = n_out = n_skip = 0
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.input, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            row = json.loads(line)
            question = row.get("question")
            label = row.get("label")
            images = row.get("images") or []
            if not isinstance(images, list):
                images = [images]
            if not question or label is None:
                n_skip += 1
                continue

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": build_user_content(images, str(question))},
            ]
            out_row = {
                "messages": messages,
                "label": label,
                "images": images,
            }
            for k in ("video", "dataset", "keyframes", "id", "video_id", "category"):
                if k in row:
                    out_row[k] = row[k]
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"[preprocess] in={n_in} out={n_out} skipped={n_skip} -> {out_path}")
    print(f"[preprocess] tools={supported}  system_len={len(system_content)} chars")


if __name__ == "__main__":
    main()
