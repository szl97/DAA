from utils import load_gsm8k, call_qwen, is_correct

data = load_gsm8k("dataset/grade-school-math/test.jsonl")

correct = 0
total = 0

for item in data[:50]:
    prompt = f"""
请计算下面的数学问题，并直接给出最终答案，格式为：#### 数字

问题：
{item['question']}
"""
    output = call_qwen(prompt)
    ok = is_correct(output, item["answer"])

    print("Correct:", ok)
    print("Model:", output)
    print("GT:", item["answer"])
    print("-" * 50)

    correct += int(ok)
    total += 1

print(f"Zero-shot Accuracy: {correct / total:.2%}")
