#!/bin/bash


cd LLaMA-Factory-main


model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct"
template=qwen
dataset="ecom_sft_ans_double_1k" # 在LLaMA-Factory-main/data/dataset_info.json下配置好
output_dir="saves/Qwen2.5-1.5B-Instruct/lora/ecom_ans_double"
num_train_epochs=30
save_steps=150
warmup_steps=100
learning_rate=5e-05
cutoff_len=340
per_device_train_batch_size=32
gradient_accumulation_steps=1
lora_rank=8
lora_alpha=16

llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path ${model_name_or_path} \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template ${template} \
    --flash_attn auto \
    --dataset_dir data \
    --dataset ${dataset} \
    --cutoff_len ${cutoff_len} \
    --learning_rate ${learning_rate} \
    --num_train_epochs ${num_train_epochs} \
    --max_samples 100000 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps ${save_steps} \
    --warmup_steps ${warmup_steps} \
    --packing False \
    --enable_thinking True \
    --report_to none \
    --output_dir ${output_dir} \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --lora_dropout 0 \
    --lora_target all




# 要合并的checkpoint列表
CHECKPOINTS=(300 450)

echo "开始批量合并checkpoint..."


for checkpoint in "${CHECKPOINTS[@]}"; do
    echo "正在合并 checkpoint-${checkpoint}..."
    
    llamafactory-cli export \
        --model_name_or_path "${model_name_or_path}" \
        --adapter_name_or_path "${output_dir}/checkpoint-${checkpoint}" \
        --template "${template}" \
        --trust_remote_code true \
        --export_dir "${output_dir}/merged_models/merged_checkpoint_${checkpoint}" \
        --export_size 5 \
        --export_device cpu \
        --export_legacy_format false
    
    if [ $? -eq 0 ]; then
        echo " checkpoint-${checkpoint} 合并成功"
    else
        echo " checkpoint-${checkpoint} 合并失败"
    fi
done

echo "合并完成!"
