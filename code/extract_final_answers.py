import re
def extract_final_answers(responses):
    """
    若response是列表则返回列表  是单条就返回单条
    """
    def _single(t: str) -> str:
        """
        优先提取 <answer>...</answer> 内的内容；
        若不存在，再退回到原来的 </think> 之后逻辑。
        """
        m = re.search(r'<answer>(.*?)</answer>', t, flags=re.DOTALL)
        if m:
            return re.sub(r'\s+', ' ', m.group(1).strip())

        idx = t.find("</think>")
        if idx != -1:
            pred = t[idx + len("</think>") :].strip()
            return re.sub(r'\s+', ' ', pred).strip()

        return t[-20:].replace('\n', ' ').strip()

    if isinstance(responses, list):
        return [_single(t) for t in responses]
    else:
        return _single(responses)

def extract_final_answers_swapped(responses):
    """
    若response是列表则返回列表  是单条就返回单条
    """
    def _single(t: str) -> str:
        """
        提取答案的优先级：
        1. 优先提取 <answer>...</answer> 内的内容
        2. 若不存在，提取 <think> 之前的内容
        3. 若 <think> 之前为空，提取 </think> 之后的内容
        4. 若 </think> 之后也为空，返回开头20个字符
        """
        # 1. 优先提取 <answer>...</answer>
        m = re.search(r'<answer>(.*?)</answer>', t, flags=re.DOTALL)
        if m:
            return re.sub(r'\s+', ' ', m.group(1).strip())

        # 2. 尝试提取 <think> 之前的内容
        think_start_idx = t.find("<think>")
        if think_start_idx != -1:
            before_think = t[:think_start_idx].strip()
            if before_think:  # 如果 <think> 之前有内容
                return re.sub(r'\s+', ' ', before_think).strip()
        
        # 3. <think> 之前为空，尝试提取 </think> 之后的内容
        # think_end_idx = t.find("</think>")
        # if think_end_idx != -1:
        #     after_think = t[think_end_idx + len("</think>"):].strip()
        #     if after_think:  # 如果 </think> 之后有内容
        #         return re.sub(r'\s+', ' ', after_think).strip()
        
        # 4. 都为空，返回开头20个字符
        return t[:20].replace('\n', ' ').strip()

    if isinstance(responses, list):
        return [_single(t) for t in responses]
    else:
        return _single(responses)


