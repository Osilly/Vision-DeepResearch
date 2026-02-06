from rllm.data.dataset import DatasetRegistry
import json
import random
import argparse

# ==================== 参数解析 ====================
def parse_args():
    parser = argparse.ArgumentParser(description="Load JSONL and register dataset to rllm")
    
    parser.add_argument(
        "--jsonl_path", 
        type=str, 
        required=True,
        help="Path to the JSONL file"
    )
    parser.add_argument(
        "--register_name", 
        type=str, 
        default="Vision-DeepResearch-QA",
        help="Name for the registered dataset (default: Vision-DeepResearch-QA)"
    )
    parser.add_argument(
        "--train_ratio", 
        type=float, 
        default=0.9,
        help="Ratio for train split (default: 0.9)"
    )
    parser.add_argument(
        "--random_seed", 
        type=int, 
        default=42,
        help="Random seed for shuffling (default: 42)"
    )
    
    return parser.parse_args()

# ==================== 读取JSONL文件 ====================
def load_jsonl(file_path):
    """读取jsonl文件，返回列表"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

# ==================== 处理并注册数据集 ====================
def process_data(data_list, start_id=0):
    """将原始数据转换为注册格式"""
    output = []
    for i, item in enumerate(data_list):
        x = dict()
        x['id'] = str(start_id + i)
        x['question'] = f"image_id: 1 \n Question: {item['question']}"
        x['answer'] = item['answer']
        x['images'] = item['images']
        output.append(x)
    return output

def register_dataset(data, register_name, split):
    """注册数据集"""
    registry_dataset = DatasetRegistry.register_dataset(register_name, data, split)
    print(f"Registered {len(data)} samples to DatasetRegistry as '{register_name}' ({split})")
    return registry_dataset

# ==================== 主流程 ====================
def main():
    # 1. 解析参数
    args = parse_args()
    
    print(f"{'='*50}")
    print(f"JSONL Path: {args.jsonl_path}")
    print(f"Register Name: {args.register_name}")
    print(f"Train Ratio: {args.train_ratio}")
    print(f"Random Seed: {args.random_seed}")
    print(f"{'='*50}\n")
    
    # 2. 加载数据
    all_data = load_jsonl(args.jsonl_path)
    print(f"Total samples loaded: {len(all_data)}")
    
    # 3. 打乱数据
    random.seed(args.random_seed)
    random.shuffle(all_data)
    
    # 4. 按比例分割
    split_idx = int(len(all_data) * args.train_ratio)
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]
    
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # 5. 处理数据
    train_output = process_data(train_data, start_id=0)
    test_output = process_data(test_data, start_id=0)
    
    # 6. 注册数据集
    register_dataset(train_output, args.register_name, "train")
    register_dataset(test_output, args.register_name, "test")
    
    print("\n✅ Dataset registration completed!")

if __name__ == "__main__":
    main()
