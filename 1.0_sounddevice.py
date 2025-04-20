'''-------开发日志-------
时间：3/27
内容：学习sounddevice的音频内容功能
目标：1.通过sounddevice录制音频
'''

import sounddevice as sd  # 音频输入/输出
from scipy.io.wavfile import write  # 保存 WAV 文件
import numpy as np  # 处理音频数据

# 参数设置
sample_rate = 44100  # 采样率（Hz）
duration = 5  # 录制时长（秒）
output_filename = "output.wav"  # 输出文件名

# 开始录音
print("开始录音（等待 5 秒）...")
recording = sd.rec(
    int(duration * sample_rate),  # 采样数 = 时长 × 采样率 就是录制的时间长度
    samplerate=sample_rate,
    channels=1,  # 单声道
    dtype="float32",  # 数据类型（兼容大多数设备）
)
sd.wait()  # 等待录制完成
print("录音结束")

# 保存为 WAV 文件
write(output_filename, sample_rate, recording)
print(f"音频已保存为: {output_filename}")

