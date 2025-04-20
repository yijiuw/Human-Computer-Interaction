'''-------开发日志-------
时间：3/31
内容：学习sounddevice和vad的结合
'''
import sounddevice as sd  # 音频输入/输出
from scipy.io.wavfile import write  # 保存 WAV 文件
from scipy.signal import resample  # 进行音频重采样
import numpy as np  # 处理音频数据
import webrtcvad
import os
import time


# 初始化 VAD
vad = webrtcvad.Vad(3)  # 激进模式（3=最严格）
target_sample_rate = 16000      # VAD 仅支持 16kHz
frame_duration = 30      # 30ms 帧（WebRTC 标准）
frame_size = int(target_sample_rate * frame_duration / 1000)  # 480 采样点

# 列出所有音频设备
input_devices = [
    (i, dev["name"])
    for i, dev in enumerate(sd.query_devices())
    if dev["max_input_channels"] > 0
]
print("可用的输入设备：")
for i, name in input_devices:
    print(f"[{i}] {name}")

#选择设备
selected_index = int(input("输入设备编号: "))
device_info = sd.query_devices(selected_index)
device_sample_rate = int(device_info["default_samplerate"])  #获取设备的采样率（Hz）
output_filename = "output.wav" # 输出文件名

# 录音参数
silence_threshold = 1.5  # 静音超时（秒）
recording = []
is_recording = False
silence_duration = 0

# 创建录音文件夹
os.makedirs("test", exist_ok=True)

def resample_audio(audio, orig_rate, target_rate):
    """将音频重采样到目标频率（用于VAD）"""
    ratio = target_rate / orig_rate
    num_samples = int(len(audio) * ratio)
    return np.interp(
        np.linspace(0, len(audio) - 1, num_samples),
        np.arange(len(audio)),
        audio
    ).astype(np.int16)

def audio_callback(indata, frames, time_info, status):
    global is_recording, recording, silence_duration

    # 转换为16-bit PCM并取单声道
    audio = (indata[:, 0] * 32767).astype(np.int16)

    # 重采样到16kHz供VAD使用
    audio_16k = resample_audio(audio, device_sample_rate, target_sample_rate)

    # VAD 检测
    is_speech = vad.is_speech(audio_16k.tobytes(), target_sample_rate)

    if is_speech:
        silence_duration = 0
        if not is_recording:
            print("\n检测到语音，开始录音...")
            is_recording = True
            recording.clear()
        recording.append(audio.copy())  # 存储处理后的 16kHz 音频
    elif is_recording:
        silence_duration += frames / device_sample_rate
        recording.append(audio.copy())
        if silence_duration >= silence_threshold:
            save_recording()


def save_recording():
    global is_recording, recording
    if recording:
        audio_data = np.concatenate(recording)
        output_filename = f"test/output_{int(time.time())}.wav"
        write(output_filename, device_sample_rate, audio_data.astype(np.int16))
        print(f"保存录音: {output_filename} ({len(audio_data) / device_sample_rate:.1f} 秒)")
    is_recording = False
    recording = []


# 4. 打开音频流
try :
    with sd.InputStream(
        device=selected_index,
        samplerate=device_sample_rate,
        channels=1,
        dtype="float32",
        blocksize=frame_size,
        callback=audio_callback

    ):
        print("等待语音输入... (按 Ctrl+C 停止)")
        sd.sleep(100000)  # 长时间运行
except KeyboardInterrupt:
    print("\n程序退出")