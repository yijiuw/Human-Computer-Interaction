from modelscope import AutoModelForCausalLM, AutoTokenizer

# 加载模型和分词器
model_name = "model/Qwen2.5-1.5B-Instruct"  # 使用官方模型名称
print("正在加载模型，请稍候...")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("模型加载完成！")
print("输入 'exit' 结束对话\n")

# 系统提示设置
system_prompt = "你叫千问，是一个18岁的女大学生，性格活泼开朗，说话俏皮，你只会用中文回答，而且回答最好不要超过50字"

# 初始化对话历史
conversation_history = [
    {"role": "system", "content": system_prompt}
]

while True:
    # 获取用户输入
    user_input = input("你: ")

    # 退出条件
    if user_input.lower() == 'exit':
        print("对话结束")
        break

    # 添加用户消息到对话历史
    conversation_history.append({"role": "user", "content": user_input})

    # 准备模型输入
    text = tokenizer.apply_chat_template(
        conversation_history,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 生成回复
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )

    # 处理输出
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids
        in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 显示回复并添加到对话历史
    print(f"AI助手: {response}\n")
    conversation_history.append({"role": "assistant", "content": response})