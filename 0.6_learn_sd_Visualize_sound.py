'''-------开发日志-------
时间：3/31
内容：学习sounddevice的实时音频流处理
'''

import sounddevice as sd
import numpy as np
import time

# 参数设置
device_info = sd.query_devices(5)
sample_rate = int(device_info["default_samplerate"]) #获取设备的采样率（Hz）
blocksize = 1024  # 每次处理的音频块大小
device = 5  # 默认设备（设为 None 自动选择）


# 回调函数：实时处理音频数据
def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"音频流状态: {status}")  # 打印错误/警告信息

    # 计算当前音频块的音量（RMS）
    volume = np.sqrt(np.mean(indata ** 2)) * 100  # 放大便于显示
    volume_bar = "|" * int(volume)  # 生成音量条

    # 打印实时音量（覆盖上一行）
    print(f"\r音量: {volume_bar} ({volume:.1f})", end="", flush=True)


#学习回调函数
def test(indata, frames, time_info, status):
    print(f"数据{indata},收到 {frames} ,时间帧{time_info}音频，状态: {status}")

# 打开音频流
try:
    print("开始监听麦克风...（按 Ctrl+C 停止）")
    with sd.InputStream(
            samplerate=sample_rate,
            channels=1,  # 单声道
            blocksize=blocksize,  # 每次处理的帧数
            device=device,  # 指定设备（None 为默认）
            dtype="float32",  # 音频数据类型
            callback=test  # 回调函数 他返回的内容就是indata, frames, time_info, status
    ):
        while True:  # 保持程序运行
            time.sleep(0.1)
except KeyboardInterrupt:
    print("\n停止监听")
except Exception as e:
    print(f"错误: {e}")