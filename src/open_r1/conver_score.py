import torch
import os
from src.open_r1.modules.annotator import Annotator
from src.open_r1.modules.tokenizer import Tokenizer
from src.open_r1.rewards import extract_predicted_answers
import argparse
from collections import Counter
from tqdm import tqdm
import torch
from collections import defaultdict
from multiprocessing import Pool
from opencc import OpenCC
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
from src.open_r1.compare_m2_for_evaluation import *
import argparse
from collections import Counter
import re
import sys
sys.setrecursionlimit(20000)

def annotate(line,sentence_to_tokenized,args):
    """
    :param line:
    :return:
    """
    sent_list = line.split("\t")[1:]
    source = sent_list[0]
    if args.segmented:
        source = source.strip()
    else:
        source = "".join(source.strip().split())
    output_str = ""
    # print(len(sent_list))
    for idx, target in enumerate(sent_list[1:]):
        try:
            source = "".join(source.strip().split())
            if not source:  # 检查是否为空
                raise ValueError("Source text is empty after processing!")

            target = "".join(target.strip().split())
            if not target:  # 检查是否为空
                raise ValueError("Target text is empty after processing!")
            if args.segmented:
                target = target.strip()
            else:
                target = "".join(target.strip().split())
            if not args.no_simplified:
                target = cc.convert(target)
            source_tokenized, target_tokenized = sentence_to_tokenized[source], sentence_to_tokenized[target]
            out, cors = annotator(source_tokenized, target_tokenized, idx)
            if idx == 0:
                output_str += "".join(out[:-1])
            else:
                output_str += "".join(out[1:-1])
        except Exception as e:
            print(f"Error in annotate: {e}")
            # 修复：继续处理而不是抛出异常
            continue
    return output_str

def parallel_to_m2(args,lines):
    m2 = []
    count = 0
    sentence_set = set()
    sentence_to_tokenized = {}
    for line in lines:
        sent_list = line.split("\t")[1:]
        # print(sent_list)
        for idx, sent in enumerate(sent_list):
            if args.segmented:
                # print(sent)
                sent = sent.strip()
            else:
                sent = "".join(sent.split()).strip()
            if idx >= 1:
                if not args.no_simplified:
                    sentence_set.add(cc.convert(sent))
                else:
                    sentence_set.add(sent)
            else:
                sentence_set.add(sent)
    # print(f"sentence_set:{sentence_set}")
    batch = []
    for sent in tqdm(sentence_set):
        count += 1
        if sent:
            batch.append(sent)
        if count % args.batch_size == 0:
            results = tokenizer(batch)
            for s, r in zip(batch, results):
                sentence_to_tokenized[s] = r  # Get tokenization map.
            batch = []
    if batch:
        results = tokenizer(batch)
        for s, r in zip(batch, results):
            sentence_to_tokenized[s] = r  # Get tokenization map.
    # print(f"sentence_to_tokenized:{sentence_to_tokenized}")
    # 单进程模式
    for line in tqdm(lines):
        # print(line)
        ret = annotate(line,sentence_to_tokenized,args)
        #rint(f"ret:{ret}")
        m2.append(ret.strip())
        # f.write(ret)
        # f.write("\n") 
    return m2


class Args:
    batch_size = 128
    device = torch.device("cuda:0")  # 自动检测GPU
    worker_num = 4                         # Jupyter中建议减少工作进程数
    granularity = "char"                   # 可选：char/word
    merge = False
    multi_cheapest_strategy = "all"        # 可选：first/all
    segmented = False                      # 是否已预分词
    no_simplified = False                  # 是否禁用简繁转换
    bpe = False                            # 是否使用BPE分词
    # 评估模式
    beta = 0.5                        # F-score权重
    # cs = True                         # 使用Span级纠错评估（默认）
    # 高级选项
    verbose = False                    # 显示详细输出
    # ilt = ["PUNCT"]                  # 过滤的错误类型
    # cat = 3                           # 错误分类展示级别
    start = None
    end = None
    max_answer_num = None
    reference_num = None
    dt = None
    ds = None
    cs = None
    cse = None
    single = None
    multi = None
    multi_hyp_avg = None
    multi_hyp_max = None
    filt = []
    cat = None


# 创建全局args实例
args = Args()
cc = OpenCC("t2s") if not args.no_simplified else None
tokenizer = Tokenizer(args.granularity, args.device, args.segmented, args.bpe)
annotator = Annotator.create_default(args.granularity, args.multi_cheapest_strategy)



    
def f_0_5_pred(preds,solution,questions):
    preds_from_completions = [p if p != '' else '...' for p in preds]
    # print(f"进入了f05的计算\n\n  其中questions:{questions} \n preds:{preds_from_completions} \n solution：{solution}\n")
    local_args = type('LocalArgs', (object,), {})()
    
    # 复制全局args的所有属性
    for attr in dir(args):
        if not attr.startswith('__'):
            setattr(local_args, attr, getattr(args, attr))
    
    assert len(preds_from_completions) == len(solution) == len(questions), \
        f"长度不一致：preds_from_completions={len(preds_from_completions)}, solutions={len(solution)}, questions={len(questions)}"
    # 整个数据集计算得分

    n = len(solution)
    data_hyp = []
    data_ref = []    
    for i in range(n):
        # 标准化文本：英文转小写（不影响中文）
        gold = solution[i].lower()
        original = questions[i].lower()
        pred = preds_from_completions[i].lower()
        data_hyp.append(str(i+1)+'\t'+original+'\t'+pred)
        data_ref.append(str(i+1)+'\t'+original+'\t'+gold)

    hyp_m2 = parallel_to_m2(local_args,data_hyp)
    ref_m2 = parallel_to_m2(local_args,data_ref)
    # Make sure they have the same number of sentences
    assert len(hyp_m2) == len(ref_m2), print(len(hyp_m2), len(ref_m2))

    # Store global corpus level best counts here
    best_dict = Counter({"tp":0, "fp":0, "fn":0})
    best_cats = {}
    # Process each sentence
    ps = []
    rs = []
    fs = []
    sents = zip(hyp_m2, ref_m2)
    for sent_id, sent in enumerate(sents):
        src = sent[0].split("\n")[0]
        hyp_edits = simplify_edits(sent[0], local_args.max_answer_num)
        ref_edits = simplify_edits(sent[1], local_args.max_answer_num)
        # Process the edits for detection/correction based on args
        hyp_dict = process_edits(hyp_edits, local_args)
        ref_dict = process_edits(ref_edits, local_args)
        if  local_args.reference_num is None or len(ref_dict.keys()) == local_args.reference_num:
            # Evaluate edits and get best TP, FP, FN hyp+ref combo.
            count_dict, cat_dict = evaluate_edits(src,
                hyp_dict, ref_dict, best_dict, sent_id, local_args)
            # Merge these dicts with best_dict and best_cats
            best_dict += Counter(count_dict)
            best_cats = merge_dict(best_cats, cat_dict)
            p,r,f = computeFScore(count_dict['tp'], count_dict['fp'], count_dict['fn'], local_args.beta)
            ps.append(p)
            rs.append(r)
            fs.append(f)
    assert len(fs) == len(preds), f"fs={len(fs)}, completions={len(preds)}"
    return {'ps':ps,'rs':rs,'fs':fs}



def f_0_5_response(response,solution,questions):
    preds_from_completions = extract_predicted_answers(response)
    preds_from_completions = [p if p != '' else '...' for p in preds_from_completions]
    # print(f"进入了f05的计算\n\n  其中questions:{questions} \n preds:{preds_from_completions} \n solution：{solution}\n")
    local_args = type('LocalArgs', (object,), {})()
    
    # 复制全局args的所有属性
    for attr in dir(args):
        if not attr.startswith('__'):
            setattr(local_args, attr, getattr(args, attr))
    
    assert len(preds_from_completions) == len(solution) == len(questions), \
        f"长度不一致：preds_from_completions={len(preds_from_completions)}, solutions={len(solutions)}, questions={len(questions)}"
    # 整个数据集计算得分

    n = len(solution)
    data_hyp = []
    data_ref = []    
    for i in range(n):
        # 标准化文本：英文转小写（不影响中文）
        gold = solution[i].lower()
        original = questions[i].lower()
        pred = preds_from_completions[i].lower()
        data_hyp.append(str(i+1)+'\t'+original+'\t'+pred)
        data_ref.append(str(i+1)+'\t'+original+'\t'+gold)

    hyp_m2 = parallel_to_m2(local_args,data_hyp)
    ref_m2 = parallel_to_m2(local_args,data_ref)
    # Make sure they have the same number of sentences
    assert len(hyp_m2) == len(ref_m2), print(len(hyp_m2), len(ref_m2))

    # Store global corpus level best counts here
    best_dict = Counter({"tp":0, "fp":0, "fn":0})
    best_cats = {}
    # Process each sentence
    ps = []
    rs = []
    fs = []
    sents = zip(hyp_m2, ref_m2)
    for sent_id, sent in enumerate(sents):
        src = sent[0].split("\n")[0]
        hyp_edits = simplify_edits(sent[0], local_args.max_answer_num)
        ref_edits = simplify_edits(sent[1], local_args.max_answer_num)
        # Process the edits for detection/correction based on args
        hyp_dict = process_edits(hyp_edits, local_args)
        ref_dict = process_edits(ref_edits, local_args)
        if  local_args.reference_num is None or len(ref_dict.keys()) == local_args.reference_num:
            # Evaluate edits and get best TP, FP, FN hyp+ref combo.
            count_dict, cat_dict = evaluate_edits(src,
                hyp_dict, ref_dict, best_dict, sent_id, local_args)
            # Merge these dicts with best_dict and best_cats
            best_dict += Counter(count_dict)
            best_cats = merge_dict(best_cats, cat_dict)
            p,r,f = computeFScore(count_dict['tp'], count_dict['fp'], count_dict['fn'], local_args.beta)
            ps.append(p)
            rs.append(r)
            fs.append(f)
    assert len(fs) == len(response), f"fs={len(fs)}, completions={len(response)}"
    return fs
            # data[sent_id]['Prec'] = p
            # data[sent_id]['Rec'] = r
            # data[sent_id]['F0.5'] = f
    # Print results
    # print_results(best_dict, best_cats, local_args)



def f_0_5(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    questions = kwargs.get('question', None)

    completion_contents = [completion[0]["content"] for completion in completions]
    preds_from_completions = extract_predicted_answers(completion_contents)
    preds_from_completions = [p[-100:] if len(p) > 100 else p for p in preds_from_completions]
    preds_from_completions = [p if p.strip() != '' else '...' for p in preds_from_completions]
    print(f"进入了f05的计算\n\n  其中questions:{questions} \n preds:{preds_from_completions} \n solution：{solution}\n")
    local_args = type('LocalArgs', (object,), {})()
    
    # 复制全局args的所有属性
    for attr in dir(args):
        if not attr.startswith('__'):
            setattr(local_args, attr, getattr(args, attr))
    
    # 使用kwargs中的值覆盖特定参数
    for key, value in kwargs.items():
        if hasattr(local_args, key):
            setattr(local_args, key, value)
    
    assert len(preds_from_completions) == len(solution) == len(questions), \
        f"长度不一致：preds_from_completions={len(preds_from_completions)}, solutions={len(solutions)}, questions={len(questions)}"
    # 整个数据集计算得分

    n = len(solution)
    data_hyp = []
    data_ref = []    
    for i in range(n):
        # 标准化文本：英文转小写（不影响中文）
        gold = solution[i].lower()
        original = questions[i].lower()
        pred = preds_from_completions[i].lower()
        data_hyp.append(str(i+1)+'\t'+original+'\t'+pred)
        data_ref.append(str(i+1)+'\t'+original+'\t'+gold)

    hyp_m2 = parallel_to_m2(local_args,data_hyp)
    ref_m2 = parallel_to_m2(local_args,data_ref)
    # Make sure they have the same number of sentences
    assert len(hyp_m2) == len(ref_m2), print(len(hyp_m2), len(ref_m2))

    # Store global corpus level best counts here
    best_dict = Counter({"tp":0, "fp":0, "fn":0})
    best_cats = {}
    # Process each sentence
    ps = []
    rs = []
    fs = []
    sents = zip(hyp_m2, ref_m2)
    for sent_id, sent in enumerate(sents):
        src = sent[0].split("\n")[0]
        hyp_edits = simplify_edits(sent[0], local_args.max_answer_num)
        ref_edits = simplify_edits(sent[1], local_args.max_answer_num)
        # Process the edits for detection/correction based on args
        hyp_dict = process_edits(hyp_edits, local_args)
        ref_dict = process_edits(ref_edits, local_args)
        if  local_args.reference_num is None or len(ref_dict.keys()) == local_args.reference_num:
            # Evaluate edits and get best TP, FP, FN hyp+ref combo.
            count_dict, cat_dict = evaluate_edits(src,
                hyp_dict, ref_dict, best_dict, sent_id, local_args)
            # Merge these dicts with best_dict and best_cats
            best_dict += Counter(count_dict)
            best_cats = merge_dict(best_cats, cat_dict)
            p,r,f = computeFScore(count_dict['tp'], count_dict['fp'], count_dict['fn'], local_args.beta)
            ps.append(p)
            rs.append(r)
            fs.append(f)
    assert len(fs) == len(completions), f"fs={len(fs)}, completions={len(completions)}"
    return fs
            # data[sent_id]['Prec'] = p
            # data[sent_id]['Rec'] = r
            # data[sent_id]['F0.5'] = f
    # Print results
    # print_results(best_dict, best_cats, local_args)

def normalize_text_for_comparison(text):
    """
    标准化文本用于比较：
    1. 转换繁体为简体
    2. 英文字母转小写
    3. 去除首尾空格
    
    Args:
        text: 输入文本
    
    Returns:
        str: 标准化后的文本
    """
    import opencc
    
    # 创建繁体转简体的转换器
    converter = opencc.OpenCC('t2s')
    
    try:
        # 转换为简体中文
        text_simplified = converter.convert(text)
        # 英文字母转小写（不影响中文）
        text_normalized = text_simplified.lower()
        return text_normalized.strip()
    except Exception as e:
        # 如果转换失败，至少转小写
        return text.lower().strip()


def if_preds_equal_question(preds, questions):
    """
    检查预测结果是否等于原问题
    先将preds和questions都转化为简体中文、英文转小写，然后比较是否相等
    
    Args:
        preds: 预测结果列表
        questions: 原问题列表
    
    Returns:
        list[bool]: 布尔值列表，True表示预测等于原问题，False表示不相等
    """
    result = []
    
    for pred, question in zip(preds, questions):
        try:
            # 标准化文本（繁转简 + 英文转小写）
            pred_normalized = normalize_text_for_comparison(pred)
            question_normalized = normalize_text_for_comparison(question)
            
            # 比较是否相等
            is_equal = pred_normalized == question_normalized
            result.append(is_equal)
            
        except Exception as e:
            # 如果处理失败，使用原文进行比较
            print(f"文本标准化失败，使用原文比较: {e}")
            is_equal = pred.strip().lower() == question.strip().lower()
            result.append(is_equal)
    
    return result




def f_0_5_check_original(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    questions = kwargs.get('question', None)
    difficulty = kwargs.get('difficulty', None)
    completion_contents = [completion[0]["content"] for completion in completions]
    preds_from_completions = extract_predicted_answers(completion_contents)
    preds_from_completions = [p[-100:] if len(p) > 100 else p for p in preds_from_completions]
    preds_from_completions = [p if p.strip() != '' else '...' for p in preds_from_completions]
    pred_equal_question = if_preds_equal_question(preds_from_completions,questions)
    # print(f"进入了f05的计算\n\n  其中questions:{questions} \n preds:{preds_from_completions} \n solution：{solution}\n")
    local_args = type('LocalArgs', (object,), {})()
    
    # 复制全局args的所有属性
    for attr in dir(args):
        if not attr.startswith('__'):
            setattr(local_args, attr, getattr(args, attr))
    
    # 使用kwargs中的值覆盖特定参数
    for key, value in kwargs.items():
        if hasattr(local_args, key):
            setattr(local_args, key, value)
    
    assert len(preds_from_completions) == len(solution) == len(questions), \
        f"长度不一致：preds_from_completions={len(preds_from_completions)}, solutions={len(solutions)}, questions={len(questions)}"
    # 整个数据集计算得分

    n = len(solution)
    data_hyp = []
    data_ref = []    
    for i in range(n):
        # 标准化文本：英文转小写（不影响中文）
        gold = solution[i].lower()
        original = questions[i].lower()
        pred = preds_from_completions[i].lower()
        data_hyp.append(str(i+1)+'\t'+original+'\t'+pred)
        data_ref.append(str(i+1)+'\t'+original+'\t'+gold)

    hyp_m2 = parallel_to_m2(local_args,data_hyp)
    ref_m2 = parallel_to_m2(local_args,data_ref)
    # Make sure they have the same number of sentences
    assert len(hyp_m2) == len(ref_m2), print(len(hyp_m2), len(ref_m2))

    # Store global corpus level best counts here
    best_dict = Counter({"tp":0, "fp":0, "fn":0})
    best_cats = {}
    # Process each sentence
    ps = []
    rs = []
    fs = []
    sents = zip(hyp_m2, ref_m2)
    for sent_id, sent in enumerate(sents):
        src = sent[0].split("\n")[0]
        hyp_edits = simplify_edits(sent[0], local_args.max_answer_num)
        ref_edits = simplify_edits(sent[1], local_args.max_answer_num)
        # Process the edits for detection/correction based on args
        hyp_dict = process_edits(hyp_edits, local_args)
        ref_dict = process_edits(ref_edits, local_args)
        if  local_args.reference_num is None or len(ref_dict.keys()) == local_args.reference_num:
            # Evaluate edits and get best TP, FP, FN hyp+ref combo.
            count_dict, cat_dict = evaluate_edits(src,
                hyp_dict, ref_dict, best_dict, sent_id, local_args)
            # Merge these dicts with best_dict and best_cats
            best_dict += Counter(count_dict)
            best_cats = merge_dict(best_cats, cat_dict)
            p,r,f = computeFScore(count_dict['tp'], count_dict['fp'], count_dict['fn'], local_args.beta)
            ps.append(p)
            rs.append(r)
            fs.append(f)
    assert len(fs) == len(completions), f"fs={len(fs)}, completions={len(completions)}"
    # fs = [f*2 if d == 'hard' and f== 1.0 else f for (f,d) in zip(fs,difficulty)]
    fs = [f if a==False else 0.0 for (f,a) in zip(fs,pred_equal_question)]
    return fs
            # data[sent_id]['Prec'] = p
            # data[sent_id]['Rec'] = r
            # data[sent_id]['F0.5'] = f
    # Print results
    # print_results(best_dict, best_cats, local_args)


