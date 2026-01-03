#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
from code.template_prompt import template_prompts_by_questions
from src.open_r1.conver_score import f_0_5_pred
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from src.open_r1.rewards import extract_predicted_answers
from code.format_reward import format_ans_double,format_ans_first,format_ans_last      
import argparse


def batch_generate(model_path = "Qwen/Qwen2.5-1.5B-Instruct",input_json="./json/ecom_new_val.json",batch_size=32,cuda=0,output_dir=None,max_new_tokens=256,sample_count=None,prompt_type="ans_double",filetype=None):
    import time
    t1 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map=f"cuda:{cuda}",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"当前Model是{model_path},device是：{model.device}")
    model.eval()

    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    with open(input_json,'r',encoding='utf-8') as f:
        data = json.load(f)
    if sample_count:
        data = data[:sample_count]

    if prompt_type == "ans_double":
        system_prompt = "你是一个中文文本错误纠正工具，可以检测和纠正文本中的错误。请检查下列文本中的错误并进行纠正，只修改错误部分并尽量保持原句结构不变，首先输出更正后的版本，然后给出你的推理过程，最后再次输出更正后的版本。请严格使用以下格式回复：<answer>（首先输出纠正后的完整文本）</answer>\n<reasoning>（简要分析错误的位置、类型和修改依据）</reasoning>\n<answer>（再次输出纠正后的完整文本）</answer>。"
    elif prompt_type == "ans_first":
        system_prompt = "你是一个中文文本错误纠正工具，可以检测和纠正文本中的错误。请检查下列文本中的错误并进行纠正，只修改错误部分并尽量保持原句结构不变，先输出更正后的版本，然后给出你的推理过程。请严格使用以下格式回复：<answer>（输出纠正后的完整文本）</answer>\n<reasoning>（简要分析错误的位置、类型和修改依据）</reasoning>。"
    elif prompt_type == "ans_last":
        system_prompt = "你是一个中文文本错误纠正工具，可以检测和纠正文本中的错误。请检查下列文本中的错误并进行纠正，只修改错误部分并尽量保持原句结构不变，给出你的推理过程，并输出更正后的版本。请严格使用以下格式回复：<reasoning>（简要分析错误的位置、类型和修改依据）</reasoning>\n<answer>（输出纠正后的完整文本）</answer>。" 
    else:
        print("prompt_type不符合要求")
        return 


    f_0_5_scores = []
    format_ansdouble = []
    format_ansfirst = []
    format_anslast = []
    accs_list = []
    time_record = []
    j=0 
    n = len(data) // batch_size  # 计算完整批次的数量
    if len(data) % batch_size != 0:
        n += 1  # 如果有剩余的数据，增加一个批次
    t2 = time.time()
    print(f"即将开始生成  已用时{t2-t1}")
    time_record = []
    for i in range(n):
        t3 = time.time()
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(data))  # 确保最后一个批次不会超出数据集范围
        batch_items = data[start_idx:end_idx]
        batch_questions = [item["original"] for item in batch_items]
        batch_solutions = [item['solution'] for item in batch_items]
        batch_prompts = template_prompts_by_questions(batch_questions,system_prompt=system_prompt)


        texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in batch_prompts]

        model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        temp = 0.0
        # 4. 批量生成
        with torch.no_grad():
            generated = model.generate(
                **model_inputs,  
                max_new_tokens=max_new_tokens,
                temperature=temp,  
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = generated[:, model_inputs['input_ids'].shape[1]:]

        responses = tokenizer.batch_decode(generated, skip_special_tokens=True)

        batch_ans_double= format_ans_double(responses)
        batch_ans_last= format_ans_last(responses)
        batch_ans_first= format_ans_first(responses)
        format_ansfirst.extend(batch_ans_first)
        format_anslast.extend(batch_ans_last)
        format_ansdouble.extend(batch_ans_double)
        t4 = time.time()
        time_record.append(t4-t3)

        preds = extract_predicted_answers(responses)

        res_dict = f_0_5_pred(preds, batch_solutions, batch_questions)
        fs = res_dict['fs']
        accs = [1 if onef==1.0 else 0 for onef in fs]
        assert len(fs) == len(batch_items) == len(preds)
        for j,item in enumerate(batch_items):
            item['eval'] = {
                'prompt':batch_prompts[j],
                "template":texts[j],
                'response':responses[j],
                'pred':preds[j],
                'f05':fs[j],
                'acc':accs[j],   
                'format_reward':{
                    'ansdouble':format_ansdouble[j],
                    'ans_first':format_ansfirst[j],
                    'ans_last':format_anslast[j],
                },
                'time':(t4-t3)/len(batch_items)
            }
        f_0_5_scores.extend(fs)
        accs_list.extend(accs)
    assert len(f_0_5_scores) == len(data)
    avg_acc = sum(accs_list) / len(accs_list) if accs_list else 0.0
    avg_time = sum(time_record) / len(data) if time_record else 0.0
    avg_f_0_5 = sum(f_0_5_scores) / len(f_0_5_scores) if f_0_5_scores else 0.0
    base_name = os.path.splitext(os.path.basename(input_json))[0]
    if not output_dir:
        output_dir = model_path+"/eval_result/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_file = output_dir+base_name+f"cate_{filetype}_prompt_{prompt_type}_max{max_new_tokens}_batch{batch_size}_sample{sample_count}_temp{temp}.json"
    with open(output_file,'w',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=2)
    print(f"数据集:{input_json} 数据集大小: {len(data)}  保存结果到: {output_file}\n 平均f_0_5: {avg_f_0_5:.4f} 平均acc: {avg_acc:.4f}  平均时间: {avg_time:.4f}")
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--input_json", type=str)
    parser.add_argument("--filetype", type=str, choices=['video', 'medical', 'ecom'])
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--prompt_type", type=str,choices=['ans_double', 'ans_first', 'ans_last'], default="ans_double")
    args = parser.parse_args()
    batch_generate(model_path=args.model_path,input_json=args.input_json,filetype=args.filetype,batch_size=args.batch_size,cuda=args.cuda,max_new_tokens=args.max_new_tokens,prompt_type=args.prompt_type)