from src.open_r1.conver_score import f_0_5_pred
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import random

def template_prompt(questions,prompt_type='ans_double'):
    if prompt_type == "ans_double":
        system_prompt = "你是一个中文文本错误纠正工具，可以检测和纠正文本中的错误。请检查下列文本中的错误并进行纠正，只修改错误部分并尽量保持原句结构不变，首先输出更正后的版本，然后给出你的推理过程，最后再次输出更正后的版本。请严格使用以下格式回复：<answer>（首先输出纠正后的完整文本）</answer>\n<reasoning>（简要分析错误的位置、类型和修改依据）</reasoning>\n<answer>（再次输出纠正后的完整文本）</answer>。"
    elif prompt_type == "ans_first":
        system_prompt = "你是一个中文文本错误纠正工具，可以检测和纠正文本中的错误。请检查下列文本中的错误并进行纠正，只修改错误部分并尽量保持原句结构不变，先输出更正后的版本，然后给出你的推理过程。请严格使用以下格式回复：<answer>（输出纠正后的完整文本）</answer>\n<reasoning>（简要分析错误的位置、类型和修改依据）</reasoning>。"
    elif prompt_type == "ans_last":
        system_prompt = "你是一个中文文本错误纠正工具，可以检测和纠正文本中的错误。请检查下列文本中的错误并进行纠正，只修改错误部分并尽量保持原句结构不变，给出你的推理过程，并输出更正后的版本。请严格使用以下格式回复：<reasoning>（简要分析错误的位置、类型和修改依据）</reasoning>\n<answer>（输出纠正后的完整文本）</answer>。" 
    else:
        print("prompt_type不符合要求")
        return 
    
    prompts = []
    for question in questions:
        prompt = []
        prompt.append({"role": "system", "content": system_prompt})
        prompt.append({"role": "user", "content": question})
        prompts.append(prompt)
    return prompts

import time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str)
parser.add_argument("--dataset_path", type=str)
parser.add_argument("--output_path", type=str)
parser.add_argument("--sample_num", type=int,default=200)
parser.add_argument("--cuda", type=int,default=0)
parser.add_argument("--prompt_type", type=str,choices=['ans_double','ans_first','ans_last'])
parser.add_argument("--num_responses", type=int,default=4)
parser.add_argument("--batch_size", type=int,default=100)
args = parser.parse_args()

t1=time.time()
NUM_RESPONSES = args.num_responses
BATCH_DATA_SIZE = args.batch_size
torch.cuda.empty_cache()
model_path = args.model_path
prompt_type = args.prompt_type
cuda = args.cuda
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map=f"cuda:{cuda}")
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.eval()


with open(args.dataset_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)
    print(f"已经加载数据集{args.dataset_path} 长度为{len(dataset)}")

dataset = dataset[:1500]
train_data = []
responses = []
prompts = []
n = len(dataset) // BATCH_DATA_SIZE  # 计算完整批次的数量
if len(dataset) % BATCH_DATA_SIZE != 0:
    n += 1  # 如果有剩余的数据，增加一个批次

t2 = time.time()
print(f"即将开始生成  已用时{t2-t1}")
for i in range(n):
    t3 = time.time()
    start_idx = i * BATCH_DATA_SIZE
    end_idx = min((i + 1) * BATCH_DATA_SIZE, len(dataset))  # 确保最后一个批次不会超出数据集范围
    batch_items = dataset[start_idx:end_idx]
    batch_questions = [item["original"] for item in batch_items]
    batch_prompts = template_prompt(batch_questions,prompt_type=prompt_type)
    prompts.extend(batch_prompts)
    texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in batch_prompts]

    # 3. 一次性 tokenize
    model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.9,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=NUM_RESPONSES,
            pad_token_id=tokenizer.eos_token_id
        )
    outputs = outputs[:, model_inputs['input_ids'].shape[1]:]

    decoded_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    responses.extend([decoded_responses[j * NUM_RESPONSES:(j + 1) * NUM_RESPONSES] for j in range(BATCH_DATA_SIZE)])
    t4 = time.time()
    assert end_idx==len(prompts)==len(responses)
    print(f"进行到了第{end_idx}/{n}条 此batch用时{t4-t3}")


from code.extract_final_answers import extract_final_answers
questions = [item['original'] for item in dataset]
solutions = [item['solution'] for item in dataset]
solutions_repeat = [sol for sol in solutions for i in range(4)]
questions_repeat = [ques for ques in questions for i in range(4)]

responses_1d = [one_res for i in range(len(responses)) for one_res in responses[i]]
preds_1d = extract_final_answers(responses_1d)
preds_2d = [preds_1d[j*4:(j+1)*4] for j in range(len(responses))]
print(len(responses))

result = f_0_5_pred(response=preds_1d,questions=questions_repeat,solution=solutions_repeat)
fs_1d = result['fs']
fs_2d = [fs_1d[j*4:(j+1)*4] for j in range(len(responses))]


labels = []

hard_data = []
easy_data = []
for i,item in enumerate(dataset):
    item['rollouts'] = {
        'prompt':prompts[i],
        'responses':responses[i],
        'preds':preds_2d[i],
        'f05s':fs_2d[i]
}


t5 = time.time()


hard_data = []
easy_data = []
for i,item in enumerate(dataset):
    one_fs = fs_2d[i]
    one_preds = preds_2d[i]
    if sum(one_fs)==0.0:
        item['rollouts']['difficulty'] = 'hard'
        hard_data.append(item)
        continue
    flag = False
    for f,p in zip(one_fs,one_preds):
        if f>0.0 and p != questions[i]:
            flag = True # 找到非原句且有效f0.5的数据 标记为easy
            item['rollouts']['difficulty'] = 'easy'
            easy_data.append(item)
            break
    if flag==False:
        item['rollouts']['difficulty'] = 'hard'
        hard_data.append(item)
        
t6 = time.time()

with open(args.output_path, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)
print(f" rollout完成，共{len(dataset)}条数据，共用时{t5-t1} easy数据有{len(easy_data)}条数据，共用时{t6-t1}  已全部存储至 {args.output_path} ")
    
    
from code.sample_by_label import sample_by_label
easy_data = sample_by_label(easy_data,args.sample_num//3+1 if args.sample_num//3 else args.sample_num//3)
easy_data = easy_data[:args.sample_num]
from code.describe_metadata import describe_metadata_bydata
describe_metadata_bydata(easy_data)
with open(args.output_easy_path, "w", encoding="utf-8") as f:
    json.dump(easy_data, f, ensure_ascii=False, indent=2)

print(f" 采样{args.sample_num}条easy数据，已保存至{args.output_easy_path}")

