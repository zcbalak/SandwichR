#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计训练数据中 instruction + input 的长度
"""

import json
import numpy as np
import matplotlib.pyplot as plt

def analyze_lengths(json_file):
    """
    分析JSON文件中instruction和input的总长度
    """
    print(f"正在读取文件: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    lengths = []
    instruction_lengths = []
    input_lengths = []
    output_lengths = []
    
    print(f"开始分析 {len(data)} 个样本...\n")
    
    for i, item in enumerate(data):
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output_text = item.get('output', '')
        
        inst_len = len(instruction)
        inp_len = len(input_text)
        out_len = len(output_text)
        total_len = inst_len + inp_len
        
        instruction_lengths.append(inst_len)
        input_lengths.append(inp_len)
        output_lengths.append(out_len)
        lengths.append(total_len)
        
        # 显示前5个样本的详细信息
        if i < 5:
            print(f"样本 {i+1}:")
            print(f"  instruction长度: {inst_len}")
            print(f"  input长度: {inp_len}")
            print(f"  output长度: {out_len}")
            print(f"  instruction + input = {total_len}")
            print()
    
    # 统计信息
    print("=" * 60)
    print("统计信息:")
    print("=" * 60)
    
    print(f"\n📊 Instruction 长度统计:")
    print(f"  最小值: {min(instruction_lengths)}")
    print(f"  最大值: {max(instruction_lengths)}")
    print(f"  平均值: {np.mean(instruction_lengths):.2f}")
    print(f"  中位数: {np.median(instruction_lengths):.2f}")
    
    print(f"\n📊 Input 长度统计:")
    print(f"  最小值: {min(input_lengths)}")
    print(f"  最大值: {max(input_lengths)}")
    print(f"  平均值: {np.mean(input_lengths):.2f}")
    print(f"  中位数: {np.median(input_lengths):.2f}")
    
    print(f"\n📊 Output 长度统计:")
    print(f"  最小值: {min(output_lengths)}")
    print(f"  最大值: {max(output_lengths)}")
    print(f"  平均值: {np.mean(output_lengths):.2f}")
    print(f"  中位数: {np.median(output_lengths):.2f}")
    
    print(f"\n📊 Instruction + Input 总长度统计:")
    print(f"  最小值: {min(lengths)}")
    print(f"  最大值: {max(lengths)}")
    print(f"  平均值: {np.mean(lengths):.2f}")
    print(f"  中位数: {np.median(lengths):.2f}")
    print(f"  标准差: {np.std(lengths):.2f}")
    
    # 百分位数
    print(f"\n📈 Instruction + Input 长度百分位数:")
    percentiles = [25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(lengths, p)
        print(f"  P{p}: {value:.2f}")
    
    # 长度分布
    print(f"\n📊 Instruction + Input 长度分布:")
    ranges = [
        (0, 100, "0-100"),
        (100, 200, "100-200"),
        (200, 300, "200-300"),
        (300, 400, "300-400"),
        (400, 500, "400-500"),
        (500, float('inf'), "500+")
    ]
    
    for min_val, max_val, label in ranges:
        count = sum(1 for l in lengths if min_val <= l < max_val)
        percentage = (count / len(lengths)) * 100
        print(f"  {label:>10}: {count:>4} 样本 ({percentage:>5.2f}%)")
    
    print("\n" + "=" * 60)
    
    # 保存详细统计到文件
    output_file = json_file.replace('.json', '_length_stats.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("详细长度统计\n")
        f.write("=" * 60 + "\n\n")
        for i, item in enumerate(data):
            inst_len = len(item.get('instruction', ''))
            inp_len = len(item.get('input', ''))
            out_len = len(item.get('output', ''))
            total_len = inst_len + inp_len
            f.write(f"样本 {i+1}: instruction={inst_len}, input={inp_len}, output={out_len}, total={total_len}\n")
    
    print(f"\n✅ 详细统计已保存到: {output_file}")
    
    return lengths, instruction_lengths, input_lengths, output_lengths

if __name__ == '__main__':
    json_file = '/home/linux/zc/LLaMA-Factory-main/data/1000_train_total_swapped_with_reasoning.json'
    lengths, inst_lens, inp_lens, out_lens = analyze_lengths(json_file)


