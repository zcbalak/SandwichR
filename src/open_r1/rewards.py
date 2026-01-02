"""Reward functions for GRPO training."""

from collections import Counter
import re
from typing import Dict

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from tqdm import tqdm
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_chinese import Rouge
import numpy as np
def extract_predicted_answers(completions):
    pattern = re.compile(r'<answer>\s*(.*?)\s*</answer>', re.DOTALL)
    return [pattern.search(c).group(1).strip()          # 取第一组捕获
            if pattern.search(c) else c.strip()                 # 没匹配到给空串
            for c in completions]



def format_ans_last(completions, **kwargs):
    pattern = r"^<reasoning>(.*?)</reasoning>\s*<answer>(.*?)</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content in completion_contents:
        # 只使用 re.DOTALL，让 . 可以匹配换行符
        # 不使用 re.MULTILINE，确保 ^ 和 \Z 匹配整个字符串的开头和结尾
        match = re.match(pattern, content.strip(), re.DOTALL)
        if match:
            # 检查 <answer></answer> 之间和 <reasoning></reasoning> 之间是否非空
            answer_content = match.group(1).strip()
            reasoning_content = match.group(2).strip()
            
            # 如果两个内容都非空，给予奖励
            if answer_content and reasoning_content:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    
    return rewards
def format_ans_first(completions, **kwargs):
    # 匹配包含 <answer> 和 <reasoning> 两个标签的格式
    # 使用 \Z 确保真正匹配到字符串末尾，而不是行末
    pattern = r"^<answer>(.*?)</answer>\s*<reasoning>(.*?)</reasoning>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content in completion_contents:
        # 只使用 re.DOTALL，让 . 可以匹配换行符s
        match = re.match(pattern, content.strip(), re.DOTALL )
        if match:
            # 检查 <answer></answer> 之间和 <reasoning></reasoning> 之间是否非空
            answer_content = match.group(1).strip()
            reasoning_content = match.group(2).strip()
            
            # 如果两个内容都非空，给予奖励
            if answer_content and reasoning_content:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    
    return rewards

def format_ans_double(completions, **kwargs):
    """Reward function that checks if the output contains two <answer> tags with identical non-empty content and non-empty <reasoning> tag."""
    pattern = r"^<answer>(.*?)</answer>\s*<reasoning>(.*?)</reasoning>\s*<answer>(.*?)</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    results = []
    
    for content in completion_contents:
        match = re.match(pattern, content, re.DOTALL)
        if match:
            # 提取三个部分的内容
            # group(1): 第一个<answer>标签中的内容
            # group(2): <reasoning>标签中的内容
            # group(3): 第二个<answer>标签中的内容
            answer1 = match.group(1).strip()  # 第一个answer的内容
            reasoning = match.group(2).strip()  # reasoning的内容
            answer2 = match.group(3).strip()  # 第二个answer的内容
            
            # 检查条件：
            # 1. 两个answer内容非空
            # 2. 两个answer内容一致
            # 3. reasoning内容非空
            if answer1 and answer2 and answer1 == answer2 and reasoning:
                results.append(1.0)
            else:
                results.append(0.0)
        else:
            results.append(0.0)
    
    return results


