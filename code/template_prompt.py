def template_prompts_by_questions(questions,system_prompt=None):
    prompts = []
    for question in questions:
        prompt = []
        prompt.append({"role": "system", "content": system_prompt})
        prompt.append({"role": "user", "content": question})
        prompts.append(prompt)
    return prompts


    if prompt_type==None:
        print("prompt_type为None，出错")
        return
    if prompt_type == "ans_double":
        system_prompt = "你是一个中文文本错误纠正工具，可以检测和纠正文本中的错误。请检查下列文本中的错误并进行纠正，只修改错误部分并尽量保持原句结构不变，首先输出更正后的版本，然后给出你的推理过程，最后再次输出更正后的版本。请严格使用以下格式回复：<answer>（首先输出纠正后的完整文本）</answer>\n<reasoning>（简要分析错误的位置、类型和修改依据）</reasoning>\n<answer>（再次输出纠正后的完整文本）</answer>。"
    elif prompt_type == "ans_first":
        system_prompt = "你是一个中文文本错误纠正工具，可以检测和纠正文本中的错误。请检查下列文本中的错误并进行纠正，只修改错误部分并尽量保持原句结构不变，先输出更正后的版本，然后给出你的推理过程。请严格使用以下格式回复：<answer>（输出纠正后的完整文本）</answer>\n<reasoning>（简要分析错误的位置、类型和修改依据）</reasoning>。"
    elif prompt_type == "ans_last":
        system_prompt = "你是一个中文文本错误纠正工具，可以检测和纠正文本中的错误。请检查下列文本中的错误并进行纠正，只修改错误部分并尽量保持原句结构不变，给出你的推理过程，并输出更正后的版本。请严格使用以下格式回复：<reasoning>（简要分析错误的位置、类型和修改依据）</reasoning>\n<answer>（输出纠正后的完整文本）</answer>。" 
    elif prompt_type == "zero_shot" or prompt_type == "no_reason":
        system_prompt = "你是一个中文文本错误纠正工具，可以检测和纠正文本中的错误。请检查下列文本中的错误并进行纠正，只修改错误部分并尽量保持原句结构不变，请直接输出更正后的版本，不做任何解释。"
    elif prompt_type == "few_shot":
        system_prompt = """你是一个中文文本错误纠正工具，可以检测和纠正文本中的错误。请检查下列文本中的错误并进行纠正，只修改错误部分并尽量保持原句结构不变，请直接输出更正后的版本，不做任何解释。下面是一些例子：
        原始文本：“管理学研究方法论第恶版”,纠正后的文本：“管理学研究方法论第二版”；
        原始文本：“大夫您好，我是一名喘哮患者，属过敏体质，”,纠正后的文本：“大夫您好，我是一名哮喘患者，属过敏体质，”；
        原始文本：“哥特堂”,纠正后的文本：“哥特教堂”。
        """
    else:
        print("prompt_type不符合要求")
        return 
    prompts = []
    for question in qs:
        prompt = []
        prompt.append({"role": "system", "content": system_prompt})
        prompt.append({"role": "user", "content": question})
        prompts.append(prompt)
    return prompts