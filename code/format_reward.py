def format_ans_double(responses):
    pattern = r"^<answer>(.*?)</answer>\s*<reasoning>(.*?)</reasoning>\s*<answer>(.*?)</answer>$"
    results = []
    import re
    for content in responses:
        match = re.match(pattern, content, re.DOTALL)
        if match:
            answer1 = match.group(1).strip()  # 第一个answer的内容
            reasoning = match.group(2).strip()  # reasoning的内容
            answer2 = match.group(3).strip()  # 第二个answer的内容
            if answer1 and answer2 and reasoning:
                results.append(1.0)
            else:
                results.append(0.0)
        else:
            results.append(0.0)
    
    return results
def format_ans_double_equal(responses):
    pattern = r"^<answer>(.*?)</answer>\s*<reasoning>(.*?)</reasoning>\s*<answer>(.*?)</answer>$"
    results = []
    import re
    for content in responses:
        match = re.match(pattern, content, re.DOTALL)
        if match:
            answer1 = match.group(1).strip()  # 第一个answer的内容
            reasoning = match.group(2).strip()  # reasoning的内容
            answer2 = match.group(3).strip()  # 第二个answer的内容
            if answer1 and answer2 and answer1==answer2 and reasoning:
                results.append(1.0)
            else:
                results.append(0.0)
        else:
            results.append(0.0)
    
    return results
def format_ans_first(responses):
    pattern = r"^<answer>(.*?)</answer>\s*<reasoning>(.*?)</reasoning>$"
    results = []
    import re
    for content in responses:
        match = re.match(pattern, content, re.DOTALL)
        if match:
            answer = match.group(1).strip()  # answer的内容
            reasoning = match.group(2).strip()  # reasoning的内容
            if answer and reasoning:
                results.append(1.0)
            else:
                results.append(0.0)
        else:
            results.append(0.0)
    return results
def format_ans_last(responses):
    pattern = r"^<reasoning>(.*?)</reasoning>\s*<answer>(.*?)</answer>$"
    results = []
    import re
    for content in responses:
        match = re.match(pattern, content, re.DOTALL)
        if match:
            reasoning = match.group(1).strip()  # reasoning的内容
            answer = match.group(2).strip()  # answer的内容
            if reasoning and answer:
                results.append(1.0)
            else:
                results.append(0.0)
        else:
            results.append(0.0)
    return results