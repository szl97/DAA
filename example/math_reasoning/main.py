import dashscope
from dashscope import Generation

dashscope.api_key = "sk-"

prompt = """
请一步一步思考并给出最终答案。

问题：
小明有 5 个苹果，吃了 2 个，又买了 3 个，现在有几个？

请先给出推理过程，再给出最终答案。
"""

response = Generation.call(
    model="qwen-plus",
    prompt=prompt,
    temperature=0.3
)

print(response.output.text)
