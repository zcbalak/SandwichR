#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess
import torch
from typing import List

from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from .evaluation import run_benchmark_jobs
from .hub import push_to_hub_revision


def is_slurm_available() -> bool:
    # returns true if a slurm queueing system is available
    try:
        subprocess.run(["sinfo"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False


class DummyConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class PushToHubRevisionCallback(TrainerCallback):
    def __init__(self, model_config) -> None:
        self.model_config = model_config

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero:
            global_step = state.global_step

            # WARNING: if you use dataclasses.replace(args, ...) the accelerator dist state will be broken, so I do this workaround
            # Also if you instantiate a new SFTConfig, the accelerator dist state will be broken
            dummy_config = DummyConfig(
                hub_model_id=args.hub_model_id,
                hub_model_revision=f"{args.hub_model_revision}-step-{global_step:09d}",
                output_dir=f"{args.output_dir}/checkpoint-{global_step}",
                system_prompt=args.system_prompt,
            )

            future = push_to_hub_revision(
                dummy_config, extra_ignore_patterns=["*.pt"]
            )  # don't push the optimizer states

            if is_slurm_available():
                dummy_config.benchmarks = args.benchmarks

                def run_benchmark_callback(_):
                    print(f"Checkpoint {global_step} pushed to hub.")
                    run_benchmark_jobs(dummy_config, self.model_config)

                future.add_done_callback(run_benchmark_callback)




CALLBACKS = {
    "push_to_hub_revision": PushToHubRevisionCallback,
}


def get_callbacks(train_config, model_config) -> List[TrainerCallback]:
    callbacks = []
    for callback_name in train_config.callbacks:
        if callback_name not in CALLBACKS:
            raise ValueError(f"Callback {callback_name} not found in CALLBACKS.")
        else:
            callbacks.append(CALLBACKS[callback_name](model_config))

    return callbacks

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的回调函数：直接保存训练过程中已存在的rollout和reward数据
"""

import json
import os
from typing import Dict, List, Any
from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


class SimpleRolloutRewardCallback(TrainerCallback):
    """
    简化的回调函数：直接保存训练过程中已存在的rollout和reward数据
    """
    
    def __init__(self, output_dir: str):
        """
        Args:
            output_dir: 保存文件的目录
        """
        self.output_dir = output_dir
        self.rollout_dir = os.path.join(output_dir, "rollouts")
        os.makedirs(self.rollout_dir, exist_ok=True)
        
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        在每个epoch结束时调用
        """
        if not state.is_world_process_zero:
            return
            
        epoch = int(state.epoch)
        global_step = state.global_step
        
        print(f"\n=== Epoch {epoch} 结束，开始保存rollout和reward数据 ===")
        
        # 获取trainer
        trainer = kwargs.get('trainer')
        if trainer is None:
            print("警告：无法获取trainer，跳过rollout保存")
            return
        
        # 尝试从trainer中获取已存在的rollout和reward数据
        self._save_existing_data(trainer, epoch, global_step)
        
        print(f"=== Epoch {epoch} rollout保存完成 ===\n")
    
    def _save_existing_data(self, trainer, epoch: int, global_step: int):
        """
        保存trainer中已存在的数据
        """
        try:
            # 方法1：尝试从trainer的日志中获取数据
            if hasattr(trainer, 'log_history') and trainer.log_history:
                self._save_from_log_history(trainer.log_history, epoch, global_step)
            
            # 方法2：尝试从trainer的内部状态获取数据
            if hasattr(trainer, 'state') and hasattr(trainer.state, 'log_history'):
                self._save_from_log_history(trainer.state.log_history, epoch, global_step)
            
            # 方法3：尝试从trainer的metrics中获取数据
            if hasattr(trainer, 'metrics'):
                self._save_from_metrics(trainer.metrics, epoch, global_step)
                
        except Exception as e:
            print(f"保存rollout数据时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_from_log_history(self, log_history: List[Dict], epoch: int, global_step: int):
        """
        从日志历史中提取并保存数据
        """
        if not log_history:
            return
            
        # 获取最新的日志条目
        latest_log = log_history[-1] if log_history else {}
        
        # 提取rollout和reward相关信息
        rollout_data = {
            'epoch': epoch,
            'global_step': global_step,
            'timestamp': latest_log.get('timestamp', ''),
            'metrics': {}
        }
        
        # 提取所有可能的reward相关指标
        reward_metrics = {}
        for key, value in latest_log.items():
            if any(keyword in key.lower() for keyword in ['reward', 'f05', 'format', 'repetition', 'edit', 'reasoning']):
                reward_metrics[key] = value
        
        rollout_data['metrics'] = reward_metrics
        
        # 保存数据
        filename = f"train_epoch_{epoch}_step_{global_step}_metrics.json"
        filepath = os.path.join(self.rollout_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(rollout_data, f, ensure_ascii=False, indent=2)
        
        print(f"已保存epoch {epoch}的metrics数据: {filepath}")
        print(f"包含的reward指标: {list(reward_metrics.keys())}")
    
    def _save_from_metrics(self, metrics: Dict, epoch: int, global_step: int):
        """
        从trainer的metrics中提取并保存数据
        """
        if not metrics:
            return
            
        rollout_data = {
            'epoch': epoch,
            'global_step': global_step,
            'metrics': metrics
        }
        
        # 保存数据
        filename = f"train_epoch_{epoch}_step_{global_step}_metrics.json"
        filepath = os.path.join(self.rollout_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(rollout_data, f, ensure_ascii=False, indent=2)
        
        print(f"已保存epoch {epoch}的metrics数据: {filepath}")
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        在每次日志记录时调用，可以实时捕获rollout和reward数据
        """
        if not state.is_world_process_zero:
            return
            
        # 获取当前日志数据
        logs = kwargs.get('logs', {})
        if not logs:
            return
        
        # 检查是否包含reward相关数据
        reward_keys = [key for key in logs.keys() if any(keyword in key.lower() for keyword in ['reward', 'f05', 'format', 'repetition', 'edit', 'reasoning'])]
        
        if reward_keys:
            # 保存实时日志数据
            epoch = int(state.epoch) if hasattr(state, 'epoch') else 0
            global_step = state.global_step
            
            log_data = {
                'epoch': epoch,
                'global_step': global_step,
                'logs': logs,
                'reward_keys': reward_keys
            }
            
            filename = f"train_epoch_{epoch}_step_{global_step}_logs.json"
            filepath = os.path.join(self.rollout_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            
            print(f"已保存step {global_step}的日志数据: {filepath}")


# 更简单的方法：直接修改训练脚本中的日志记录
class RolloutDataLogger:
    """
    直接在训练过程中记录rollout和reward数据的类
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.rollout_dir = os.path.join(output_dir, "rollouts")
        os.makedirs(self.rollout_dir, exist_ok=True)
        self.epoch_data = {}
    
    def log_rollout_data(self, epoch: int, global_step: int, rollout_data: Dict, reward_scores: List[float]):
        """
        记录rollout和reward数据
        """
        if epoch not in self.epoch_data:
            self.epoch_data[epoch] = []
        
        data_entry = {
            'global_step': global_step,
            'rollout_data': rollout_data,
            'reward_scores': reward_scores
        }
        
        self.epoch_data[epoch].append(data_entry)
    
    def save_epoch_data(self, epoch: int):
        """
        保存整个epoch的数据
        """
        if epoch not in self.epoch_data:
            return
            
        filename = f"train_epoch_{epoch}_rollout_rewards.json"
        filepath = os.path.join(self.rollout_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.epoch_data[epoch], f, ensure_ascii=False, indent=2)
        
        print(f"已保存epoch {epoch}的完整rollout和reward数据: {filepath}")
        
        # 清理内存
        del self.epoch_data[epoch]


class EpochRewardEvaluationCallback(TrainerCallback):
    """
    在每个epoch结束后在训练集和测试集上生成响应并计算所有reward_func得分的callback
    """
    
    def __init__(self, eval_dataset=None, test_dataset=None, output_dir=None, 
                 reward_funcs=None, temperature=0.7, max_new_tokens=512):
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.output_dir = output_dir
        self.reward_funcs = reward_funcs or []
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.eval_results_dir = os.path.join(output_dir, "epoch_reward_evaluations")
        os.makedirs(self.eval_results_dir, exist_ok=True)
        self._trainer_ref = None  # 保存 trainer 引用
    
    def on_train_begin(self, args, state, control, **kwargs):
        """在训练开始时获取 trainer 引用"""
        if 'model' in kwargs:
            # 某些版本传递的是 model 而不是 trainer
            # 我们可以从其他地方获取需要的信息
            self._model = kwargs.get('model')
            self._tokenizer = kwargs.get('tokenizer') or kwargs.get('processing_class')
        
        # 注意：由于 transformers 的 callback 机制，trainer 通常不会传递到 kwargs
        # 我们需要通过其他方式获取
        
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        在每个epoch结束时进行reward评测
        """
        if not state.is_world_process_zero:
            return
            
        epoch = int(state.epoch)
        global_step = state.global_step
        
        print(f"\n=== Epoch {epoch} Reward评测开始 (Step {global_step}) ===")
        
        # 尝试多种方式获取 trainer
        trainer = kwargs.get('trainer')
        
        if trainer is None:
            # 方法2: 从 model 参数获取（某些版本）
            model = kwargs.get('model')
            if model is not None:
                print(f"  从 kwargs 中获取到 model，但没有 trainer")
                print(f"  ⚠️  无法进行评测（需要 trainer 对象）")
                print(f"  建议：在较新版本的 transformers 中，可能需要修改 callback 实现")
                return
            else:
                print(f"Warning: Trainer and model not found in kwargs, skipping evaluation")
                print(f"  可用的 kwargs 键: {list(kwargs.keys())}")
                return
            
        # # 评测训练集
        # train_results = {}
        # if hasattr(trainer, 'train_dataset') and trainer.train_dataset is not None:
        #     print("正在评测训练集...")
        #     try:
        #         train_results = self._evaluate_dataset_rewards(trainer, trainer.train_dataset, "train", epoch)
        #         print(f"训练集Reward评测完成")
        #     except Exception as e:
        #         print(f"训练集Reward评测失败: {e}")
        #         train_results = {"error": str(e)}
        
        # 评测测试集
        test_results = {}
        if self.test_dataset is not None:
            print("正在评测测试集...")
            try:
                test_results = self._evaluate_dataset_rewards(trainer, self.test_dataset, "test", epoch)
                print(f"测试集Reward评测完成")
            except Exception as e:
                print(f"测试集Reward评测失败: {e}")
                test_results = {"error": str(e)}
        
        # 计算平均得分
        # train_avg_scores = self._calculate_average_scores(train_results)
        test_avg_scores = self._calculate_average_scores(test_results)
        
        # 保存评测结果
        evaluation_results = {
            "epoch": epoch,
            "global_step": global_step,
            # "train_results": train_results,
            # "train_avg_scores": train_avg_scores,
            "test_results": test_results,
            "test_avg_scores": test_avg_scores,
            "timestamp": state.log_history[-1] if state.log_history else {}
        }
        
        # 保存到文件
        eval_file = os.path.join(self.eval_results_dir, f"epoch_{epoch}_reward_evaluation.json")
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        
        print(f"Reward评测结果已保存到: {eval_file}")
        # print(f"训练集平均得分: {train_avg_scores}")
        print(f"测试集平均得分: {test_avg_scores}")
        print(f"=== Epoch {epoch} Reward评测结束 ===\n")
        
        # 记录到wandb或其他日志系统
        if hasattr(trainer, 'log'):
            log_metrics = {}
            # for reward_name, score in train_avg_scores.items():
            #     log_metrics[f"epoch_{epoch}_train_{reward_name}_mean"] = score
            for reward_name, score in test_avg_scores.items():
                log_metrics[f"epoch_{epoch}_test_{reward_name}_mean"] = score
            trainer.log(log_metrics)
        
        return control
    
    def _evaluate_dataset_rewards(self, trainer, dataset, dataset_name, epoch):
        """
        在指定数据集上生成响应并计算所有reward_func得分
        """
        total_samples = len(dataset)
        print(f"开始评测{dataset_name}数据集，共{total_samples}个样本...")
        
        # 存储所有样本的详细结果
        all_sample_results = []
        
        for i, example in enumerate(dataset):
            if i % 10 == 0:
                print(f"处理进度: {i}/{total_samples}")
            
            try:
                # 获取prompt和solution
                prompt = example["prompt"]
                solution = example["solution"]
                question = example.get("question", "")
                
                # 生成响应（每个样本只生成一个响应）
                inputs = trainer.tokenizer.apply_chat_template(
                    prompt, 
                    tokenize=True, 
                    add_generation_prompt=True, 
                    return_tensors="pt"
                ).to(trainer.model.device)
                
                with torch.no_grad():
                    outputs = trainer.model.generate(
                        inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        do_sample=True,
                        pad_token_id=trainer.tokenizer.eos_token_id
                    )
                
                # 解码响应
                generated_response = trainer.tokenizer.decode(
                    outputs[0][len(inputs[0]):], 
                    skip_special_tokens=True
                )
                
                # 计算每个reward_func的得分
                sample_reward_scores = {}
                for reward_func in self.reward_funcs:
                    try:
                        # 计算该reward_func对单个响应的得分
                        reward_scores = reward_func([generated_response], solution=solution, question=question)
                        sample_reward_scores[reward_func.__name__] = reward_scores[0]  # 取第一个（也是唯一一个）得分
                    except Exception as e:
                        print(f"计算reward {reward_func.__name__} 时出错: {e}")
                        sample_reward_scores[reward_func.__name__] = 0.0
                
                # 保存该样本的详细结果
                sample_result = {
                    "sample_index": i,
                    "prompt": prompt,
                    "solution": solution,
                    "question": question,
                    "generated_response": generated_response,
                    "reward_scores": sample_reward_scores
                }
                all_sample_results.append(sample_result)
                
            except Exception as e:
                print(f"处理样本{i}时出错: {e}")
                # 添加错误样本的结果
                sample_result = {
                    "sample_index": i,
                    "error": str(e),
                    "reward_scores": {reward_func.__name__: 0.0 for reward_func in self.reward_funcs}
                }
                all_sample_results.append(sample_result)
        
        print(f"{dataset_name}数据集Reward评测完成，共处理{len(all_sample_results)}个样本")
        
        # 保存详细结果到文件
        detailed_file = os.path.join(self.eval_results_dir, f"epoch_{epoch}_{dataset_name}_detailed_results.json")
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(all_sample_results, f, ensure_ascii=False, indent=2)
        
        return all_sample_results
    
    def _calculate_average_scores(self, results):
        """
        计算各个reward_func的平均得分
        """
        if not results or "error" in results:
            return {}
        
        avg_scores = {}
        
        # 获取所有reward_func名称
        if results:
            first_sample = results[0]
            if "reward_scores" in first_sample:
                reward_names = first_sample["reward_scores"].keys()
                
                for reward_name in reward_names:
                    all_scores = []
                    for sample in results:
                        if "reward_scores" in sample and reward_name in sample["reward_scores"]:
                            # 直接取该样本的得分（已经是单个值）
                            score = sample["reward_scores"][reward_name]
                            all_scores.append(score)
                    
                    if all_scores:
                        avg_scores[reward_name] = sum(all_scores) / len(all_scores)
                    else:
                        avg_scores[reward_name] = 0.0
        
        return avg_scores


# 在callbacks.py中添加
def get_simple_rollout_callback(output_dir: str):
    """
    获取简化的rollout保存回调函数
    """
    return SimpleRolloutRewardCallback(output_dir)