'''-------开发日志-------
时间：3/27
内容：学习sounddevice的音频内容功能
目标：1.指定设备录制音频
'''

import sounddevice as sd  # 音频输入/输出
from scipy.io.wavfile import write  # 保存 WAV 文件
import numpy as np  # 处理音频数据

# 列出所有音频设备
input_devices = []
for i, dev in enumerate(sd.query_devices()):
    if dev["max_input_channels"] > 0:
        input_devices.append((i,dev["name"]))
'''
特殊写法
input_devices = 
[
    (i, dev["name"])
    for i, dev in enumerate(sd.query_devices())
    if dev["max_input_channels"] > 0
]
'''
#列出设备
print("可用的输入设备：")
for i, name in input_devices:
    print(f"[{i}] {name}")

# 2. 选择设备
selected_index = int(input("输入设备编号: "))
device_info = sd.query_devices(selected_index)

# 参数设置
sample_rate = int(device_info["default_samplerate"]) #获取设备的采样率（Hz）
duration = 5  # 录制时长（秒）
output_filename = "output.wav"  # 输出文件名

# 3. 录制音频
print(f"使用设备: {device_info['name']}, 采样率: {sample_rate} Hz")
recording = sd.rec(
    int(duration * sample_rate),  # 采样数 = 时长 × 采样率 就是录制的时间长度
    samplerate=sample_rate,
    channels=1,  # 单声道
    device=selected_index, #使用选择的设备
    dtype="float32",  # 数据类型（兼容大多数设备）
)
sd.wait()  # 等待录制完成
print("录音结束")

# 保存为 WAV 文件
write(output_filename, sample_rate, recording)
print(f"音频已保存为: {output_filename}")

