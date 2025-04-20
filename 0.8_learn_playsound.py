"""-------开发日志-------
时间：3/31
内容：学习edgeTTS的基本功能
"""

from playsound import playsound
import edge_tts
import asyncio


async def play_speech():
    voice = "zh-CN-XiaoxiaoNeural"  # 中文女声"晓晓"
    text = "正在播放语音测试。"
    output_file = "output.mp3"

    communicate = edge_tts.Communicate(text, voice)
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)
    playsound("output.mp3")


asyncio.run(play_speech())