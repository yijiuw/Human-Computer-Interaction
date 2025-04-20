'''-------开发日志-------
时间：4/8
内容：学习edgeTTS的基本功能
'''

import edge_tts
import asyncio


async def generate_speech():
    voice = "zh-CN-YunxiNeural"  # 中文男声"云健"
    text = "欢迎使用Edge-TTS，这是一个免费的文字转语音工具。"
    output_file = "output.mp3"

    # 生成语音并保存
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)
    print(f"语音已保存到 {output_file}")


asyncio.run(generate_speech())