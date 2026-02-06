#!/usr/bin/env python3
"""
Convert parquet file to jsonl file, extracting image bytes to image files
"""

import pandas as pd
import json
import os
from pathlib import Path
from PIL import Image
import argparse
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if pd.isna(obj):
            return None
        return super().default(obj)


def convert_to_serializable(obj):
    """Recursively convert non-serializable objects"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj


def bytes_dict_to_bytes(byte_data):
    """
    Convert various byte data formats to bytes
    Supports: dict ({"0": 137, "1": 80, ...}), list, bytes, ndarray
    """
    if isinstance(byte_data, bytes):
        return byte_data
    elif isinstance(byte_data, np.ndarray):
        return bytes(byte_data.tolist())
    elif isinstance(byte_data, (list, tuple)):
        return bytes(byte_data)
    elif isinstance(byte_data, dict):
        # Sort by key index
        max_index = max(int(k) for k in byte_data.keys())
        byte_array = bytearray(max_index + 1)
        for k, v in byte_data.items():
            byte_array[int(k)] = v
        return bytes(byte_array)
    else:
        raise ValueError(f"Unsupported byte data type: {type(byte_data)}")


def get_image_extension(img_bytes):
    """Determine image format based on magic number"""
    if img_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        return 'png'
    elif img_bytes[:2] == b'\xff\xd8':
        return 'jpg'
    elif img_bytes[:6] in (b'GIF87a', b'GIF89a'):
        return 'gif'
    elif img_bytes[:4] == b'RIFF' and img_bytes[8:12] == b'WEBP':
        return 'webp'
    elif img_bytes[:2] == b'BM':
        return 'bmp'
    else:
        return 'png'  # default


def convert_parquet_to_jsonl(parquet_path, output_jsonl_path, image_output_dir):
    """
    Convert parquet file to jsonl file, saving image bytes as image files
    
    Args:
        parquet_path: Path to input parquet file
        output_jsonl_path: Path to output jsonl file
        image_output_dir: Directory to save extracted images
    """
    # Create image output directory
    image_output_dir = Path(image_output_dir).resolve()
    image_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read parquet file
    print(f"Reading parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"Total rows: {len(df)}")
    
    img_count = 0
    
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for row_idx, row in df.iterrows():
            row_data = row.to_dict()
            
            # Process images field
            if 'images' in row_data and row_data['images'] is not None:
                image_paths = []
                images = row_data['images']
                
                # Ensure images is a list
                if isinstance(images, np.ndarray):
                    images = images.tolist()
                if not isinstance(images, list):
                    images = [images]
                
                for img_idx, img_data in enumerate(images):
                    try:
                        # Convert dict/list byte data to bytes
                        img_bytes = bytes_dict_to_bytes(img_data)
                        
                        # Get image extension
                        ext = get_image_extension(img_bytes)
                        
                        # Generate image filename
                        img_filename = f"{row_idx:06d}_{img_idx:02d}.{ext}"
                        img_path = image_output_dir / img_filename
                        
                        # Save image
                        with open(img_path, 'wb') as img_f:
                            img_f.write(img_bytes)
                        
                        # Verify image validity (optional)
                        try:
                            Image.open(img_path).verify()
                        except Exception as e:
                            print(f"Warning: Image verification failed for {img_path}: {e}")
                        
                        # Add absolute path
                        image_paths.append(str(img_path))
                        img_count += 1
                        
                    except Exception as e:
                        print(f"Error processing image at row {row_idx}, image {img_idx}: {e}")
                        continue
                
                row_data['images'] = image_paths
            
            # Convert all numpy types to native Python types
            row_data = convert_to_serializable(row_data)
            
            # Write to jsonl
            f.write(json.dumps(row_data, ensure_ascii=False, cls=NumpyEncoder) + '\n')
            
            # Progress display
            if (row_idx + 1) % 100 == 0:
                print(f"Processed {row_idx + 1} rows...")
    
    print(f"\n{'='*50}")
    print(f"Conversion complete!")
    print(f"JSONL saved to: {output_jsonl_path}")
    print(f"Images saved to: {image_output_dir}")
    print(f"Total images extracted: {img_count}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert parquet to jsonl with image extraction'
    )
    parser.add_argument('--parquet_path', type=str, required=True,
                        help='Path to input parquet file')
    parser.add_argument('--output_jsonl', type=str, required=True,
                        help='Path to output jsonl file')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory to save extracted images')
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.parquet_path):
        print(f"Error: Parquet file not found: {args.parquet_path}")
        return 1
    
    convert_parquet_to_jsonl(args.parquet_path, args.output_jsonl, args.image_dir)
    return 0


if __name__ == "__main__":
    exit(main())
