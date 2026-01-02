from datasets import Dataset, DatasetDict
import json, os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--test", type=str)
parser.add_argument("--train", type=str)
parser.add_argument("--out_path", type=str)
args = parser.parse_args()
test = args.test
train = args.train
out_path = args.out_path


def make_dataset(file_path):
    """
    创建数据集，保留difficulty标签
    """
    dataset = {
        "messages": [],
        "difficulty": [],  # 新增：存储difficulty标签,
        "error_label":[]
    }
    
    with open(file_path, "r") as f:
        data_all = json.load(f)
    # describe_metadata_bydata(data_all)
    for i in range(len(data_all)):
        data = data_all[i]
        
        # 创建消息格式
        message = [
            {"role": "user", "content": data['original']},
            {"role": "assistant", "content": data['solution']}
        ]
        dataset["messages"].append(message)
        # print(data)
        if 'rollouts' in data:
        # 添加difficulty标签（从JSON中的label字段获取）
            difficulty = data['rollouts'].get('difficulty', 'unknown')  # 如果label字段不存在，默认为'unknown'
        else:
            difficulty = data.get('label', 'unknown') 
        dataset["error_label"].append(data['metadata']['label'])
        dataset["difficulty"].append(difficulty)
    
    return dataset

train_dataset = make_dataset(train)
train_dataset = Dataset.from_dict(train_dataset)
print(train_dataset)
test_dataset = make_dataset(test)
test_dataset = Dataset.from_dict(test_dataset)
print(test_dataset)

dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})
  
dataset.save_to_disk(out_path)
print(f" train:{train}\n test:{test}\n Dataset saved to {out_path}")

