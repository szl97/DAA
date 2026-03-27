from utils import load_gsm8k, call_qwen, is_correct

data = load_gsm8k("dataset/grade-school-math/test.jsonl")

correct = 0
total = 0

for item in data[:50]:
    prompt = f"""
请一步一步思考并解决下面的数学问题。
最后一行必须是：#### 数字

问题：
{item['question']}
"""
    output = call_qwen(prompt)
    ok = is_correct(output, item["answer"])

    print("Correct:", ok)
    print(output)
    print("GT:", item["answer"])
    print("-" * 50)

    correct += int(ok)
    total += 1

print(f"CoT Accuracy: {correct / total:.2%}")
