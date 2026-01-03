# Sandwich Reasoning: An Answer-Reasoning-Answer Approach for Low-Latency Query Correction

This is the official implementation code of the paper "Sandwich Reasoning: An Answer-Reasoning-Answer Approach for Low-Latency Query Correction".

## Project structure

```
SandwichR/
├── code/                          # Data processing and tool scripts
│   ├── reject_sampling.py        # reject sampling
│   ├── create_dataset.py          # dataset conversion, json->dataset
│   ├── format_reward.py          # format reward
│   ├── template_prompt.py        # prompt template
│   └── ...
├── data/                         # Three dataset directories
│   ├── ecom                      
│   ├── medical       
│   └── video         
├── src/                           # Core source code
│   └── open_r1/                  # Training and evaluating code
│       ├── rag_grpo_stage2.py    # GRPO training main script
│       ├── rewards.py            # Reward function
│       ├── conver_score.py       # F0.5 evaluation metric
│       └── ...
├── recipes/                       # training profiles
│   └── Qwen2.5-Qwen-1.5B-Instrcut/  # Model profile
├── datasets/                      # dataset dataset
├── SFT.sh                        # SFT Training script
├── GRPO.sh                      # GRPO Reinforcement Learning Training Script
├── inference.py                   # Inference evaluation code
└── readme.md                      
```

## requirements

- Python >= 3.8
- CUDA >= 11.4 
- PyTorch >= 2.0.0


## Quick Start

```
cd SandwichR

# 1. SFT training
bash SFT.sh

# 2. Reject sampling and prepare the dataset for RL training
python code/reject_sampling.py \
    --model_path ./sft_model \
    --dataset_path data/ecom/train/train_data.json \
    --output_path output/rollout.json \
    --output_easy_path output/easy_200.json \
    --sample_num 200 \
    --prompt_type ans_double

# 3. Dataset transformation, converting JSON format data to HuggingFace datasets format.
python code/create_dataset.py \
    --train output/easy_200.json \
    --test data/ecom/dev/dev_1k.json \
    --out_path datasets/rl_dataset

# 4. RL training
# Modify the configuration in recipes/Qwen2.5-Qwen-1.5B-Instrcut/ans_double.yaml
bash GRPO.sh

# 5. Inference evaluation
python inference.py \
    --model_path output/grpo_model \
    --input_json data/ecom/dev/dev_1k.json \
    --filetype ecom \
    --prompt_type ans_double
```


## Reference
The SandwichR is built based on the following project:
- [openr1](https://github.com/huggingface/open-r1)
- [LlamaFactory](https://github.com/hiyouga/LlamaFactory)