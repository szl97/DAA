import json
import re
import dashscope
from dashscope import Generation

dashscope.api_key = "sk-"


def load_gsm8k(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def call_qwen(prompt):
    response = Generation.call(
        model="qwen-plus",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return response.output.text


def extract_answer(text):
    """
    提取模型最终数值答案
    优先匹配 #### 数字
    否则退化为最后一个数字
    """
    match = re.search(r"####\s*([-+]?\d+\.?\d*)", text)
    if match:
        return match.group(1)

    numbers = re.findall(r"[-+]?\d+\.?\d*", text)
    return numbers[-1] if numbers else None


def is_correct(model_output, gt_answer):
    pred = extract_answer(model_output)
    gt = extract_answer(gt_answer)
    return pred == gt
