from utils import load_gsm8k, call_qwen, is_correct

data = load_gsm8k("dataset/grade-school-math/test.jsonl")

FEW_SHOT = (
    "示例1：\n"
    "问题：Tom has 3 apples and eats 1. How many apples remain?\n"
    "答案：#### 2\n\n"
    "示例2：\n"
    "问题：A box has 4 pencils and 6 more are added. How many pencils now?\n"
    "答案：#### 10\n\n"
)

correct = 0
total = 0

for item in data[:10]:
    prompt = (
        FEW_SHOT +
        "请回答下面的问题，给出最终答案，格式为：#### 数字\n\n"
        f"问题：{item['question']}\n"
    )

    output = call_qwen(prompt)
    ok = is_correct(output, item["answer"])

    print("Correct:", ok)
    print("Model:", output)
    print("GT:", item["answer"])
    print("-" * 50)

    correct += int(ok)
    total += 1

print(f"Few-shot Accuracy: {correct / total:.2%}")
