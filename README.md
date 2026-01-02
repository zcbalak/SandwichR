# Sandwich Reasoning: An Answer-Reasoning-Answer Approach for Low-Latency Query Correction

这是论文 "Sandwich Reasoning: An Answer-Reasoning-Answer Approach for Low-Latency Query Correction" 的官方实现代码。


## 项目结构

```
SandwichR/
├── code/                          # 数据处理和工具脚本
│   ├── reject_sampling.py        # reject sampling
│   ├── create_dataset.py          # 数据集转换,json->dataset
│   ├── format_reward.py          # 格式奖励
│   ├── template_prompt.py        # prompt模板
│   └── ...
├── data/                         # 三个数据集目录
│   ├── ecom                      
│   ├── medical       
│   └── video         
├── src/                           # 核心源代码
│   └── open_r1/                  # 训练和评估代码
│       ├── rag_grpo_stage2.py    # GRPO训练主脚本
│       ├── rewards.py            # 奖励函数
│       ├── conver_score.py       # F0.5评估指标
│       └── ...
├── recipes/                       # 训练配置文件
│   └── Qwen2.5-Qwen-1.5B-Instrcut/  # 模型配置文件
├── datasets/                      # dataset数据集
├── SFT.sh                        # SFT训练脚本
├── GRPO.sh                      # GRPO强化学习训练脚本
├── inference.py                   # 推理评估代码
└── readme.md                      # 本文档
```

## 环境要求

- Python >= 3.8
- CUDA >= 11.4 (推荐)
- PyTorch >= 2.0.0


## 快速开始示例

```
cd SandwichR

# 1. SFT 训练（使用 LLaMA-Factory）
bash SFT.sh

# 2. 拒绝采样，为RL训练准备数据集
python code/reject_sampling.py \
    --model_path ./sft_model \
    --dataset_path data/ecom/train/train_data.json \
    --output_path output/rollout.json \
    --output_easy_path output/easy_200.json \
    --sample_num 200 \
    --prompt_type ans_double

# 3. 数据集转换,将 JSON 格式的数据转换为 HuggingFace datasets 格式。
python code/create_dataset.py \
    --train output/easy_200.json \
    --test data/ecom/dev/dev_1k.json \
    --out_path datasets/rl_dataset

# 4. RL 训练
# 修改 recipes/Qwen2.5-Qwen-1.5B-Instrcut/ans_double.yaml 中的配置
bash GRPO.sh

# 5. 推理评估
python inference.py \
    --model_path output/grpo_model \
    --input_json data/ecom/dev/dev_1k.json \
    --filetype ecom \
    --prompt_type ans_double
```

=======
# SandwichR
>>>>>>> 02f894942b5b997d3cb2532cf6569646d2fbb648
