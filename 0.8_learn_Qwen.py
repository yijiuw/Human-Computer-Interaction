"""-------开发日志-------
时间：4/9
内容：学习Qwen的基本用法
"""

# 导入ModelScope库中的模型和分词器自动加载类
from modelscope import AutoModelForCausalLM, AutoTokenizer

# 指定模型名称/路径（Qwen2.5-1.5B-Instruct是一个15亿参数的中英双语对话模型）
model_name = "model/Qwen2.5-0.5B-Instruct"

# 加载预训练模型
# torch_dtype="auto" 自动选择合适的数据类型(float16/float32)
# device_map="auto" 自动选择运行设备(优先使用GPU，如果没有则用CPU)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# 加载与模型配套的分词器(负责文本和token之间的转换)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 用户的问题/提示
prompt = "请简单介绍一下大型语言模型."

# 构造对话历史消息列表
# 每条消息需要指定角色(role)和内容(content)
messages = [
    # system消息设置AI助手的角色和身份
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    # user消息是用户的实际提问
    {"role": "user", "content": prompt}
]

# 将对话历史转换为模型可以理解的格式
# tokenize=False 表示只格式化不立即分词
# add_generation_prompt=True 会添加特殊token告诉模型该生成回复了
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
"""
# 对格式化后的文本进行分词，并转换为模型输入格式
# return_tensors="pt" 返回PyTorch张量
# .to(model.device) 确保输入数据与模型在同一设备上(CPU/GPU)
"""
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 使用模型生成回复
# max_new_tokens=512 限制生成的最大token数量(防止生成过长)
generated_ids = model.generate(
    **model_inputs,  # 传入分词后的输入
    max_new_tokens=512
)
print(generated_ids)

# 处理生成结果：从输出中去掉输入部分，只保留新生成的token
generated_ids = [
    output_ids[len(input_ids):]  # 取输出中超出输入长度的部分
    for input_ids, output_ids
    in zip(model_inputs.input_ids, generated_ids)
]

# 将生成的token IDs解码为人类可读的文本
# skip_special_tokens=True 跳过特殊token(如结束符等)
# [0] 取batch中的第一个结果(本例中batch_size=1)
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(response)

# 最终response变量中就是模型生成的回答