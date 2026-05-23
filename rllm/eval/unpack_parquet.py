#!/usr/bin/env python3
"""
Extract images embedded in a packed parquet back to local files,
rewrite the 'images' column and any filename references in 'question'
with the resulting local absolute paths, and write a ready-to-infer parquet.

Usage:
    python unpack_parquet.py --output_dir ./data/ready \
                             --images_dir ./data/images \
                             ./hf_data/data/*.parquet
"""

import argparse
import base64
import json
import os

import pandas as pd


def unpack_one(input_path: str, output_path: str, images_dir: str) -> None:
    print(f"[unpack] {input_path}")
    df = pd.read_parquet(input_path)

    if "image_packed" not in df.columns:
        print("  No image_packed column, copying as-is.")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df.to_parquet(output_path, index=False)
        return

    os.makedirs(images_dir, exist_ok=True)

    has_question = "question" in df.columns
    new_images = []
    new_questions = list(df["question"].astype(str)) if has_question else None

    for enum_i, (row_idx, row) in enumerate(df.iterrows()):
        packed_str = row.get("image_packed", "[]") or "[]"
        try:
            packed = json.loads(packed_str)
        except Exception:
            packed = []

        local_paths = []
        for i, item in enumerate(packed):
            b64 = item.get("data")
            filename = item["filename"]
            # row_idx + 序号 + 文件名，避免不同数据集同名冲突
            fname = f"{row_idx}_{i}_{filename}"
            out_path = os.path.join(images_dir, fname)

            if b64 is None:
                local_paths.append(None)
                continue

            if not os.path.exists(out_path):
                with open(out_path, "wb") as f:
                    f.write(base64.b64decode(b64))

            abs_out = os.path.abspath(out_path)
            local_paths.append(abs_out)

            # 将 question 里出现的原文件名替换为本地绝对路径
            if new_questions is not None:
                new_questions[enum_i] = new_questions[enum_i].replace(filename, abs_out)

        new_images.append(local_paths)

    df["images"] = new_images
    if new_questions is not None:
        df["question"] = new_questions
    df = df.drop(columns=["image_packed"])

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"  -> {output_path}  (rows={len(df)})\n")


def main():
    parser = argparse.ArgumentParser(
        description="Unpack image-packed parquet into a ready-to-infer parquet."
    )
    parser.add_argument("inputs", nargs="+", help="packed parquet paths")
    parser.add_argument("--output_dir", default="./data/ready",
                        help="directory for the unpacked parquet files")
    parser.add_argument("--images_dir", default="./data/images",
                        help="directory to write extracted image files into")
    args = parser.parse_args()

    for p in args.inputs:
        unpack_one(
            p,
            os.path.join(args.output_dir, os.path.basename(p)),
            args.images_dir,
        )
    print("All done.")


if __name__ == "__main__":
    main()
