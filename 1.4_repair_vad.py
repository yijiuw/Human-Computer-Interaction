'''-------开发日志-------
时间：3/31
内容：修复VAD音频处理问题
'''
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import webrtcvad
import os
import time

# 初始化 VAD
vad = webrtcvad.Vad(3)  # 激进模式（3=最严格）
target_sample_rate = 16000  # VAD 仅支持 16kHz
frame_duration = 30  # 30ms 帧（WebRTC 标准）
frame_size = int(target_sample_rate * frame_duration / 1000)  # 480 采样点

# 列出所有音频设备
input_devices = [
    (i, dev["name"], dev["default_samplerate"])
    for i, dev in enumerate(sd.query_devices())
    if dev["max_input_channels"] > 0
]
print("可用的输入设备：")
for i, name, rate in input_devices:
    print(f"[{i}] {name} (采样率: {rate}Hz)")

# 选择设备
selected_index = int(input("输入设备编号: "))
device_info = sd.query_devices(selected_index)
device_sample_rate = int(device_info["default_samplerate"])

# 录音参数
silence_threshold = 1.5  # 静音超时（秒）
recording = []
is_recording = False
silence_duration = 0

# 创建录音文件夹
os.makedirs("recordings", exist_ok=True)


def prepare_audio_for_vad(audio, orig_rate):
    """准备符合VAD要求的音频数据"""
    # 确保是单声道
    if len(audio.shape) > 1:
        audio = audio[:, 0]

    # 转换为16-bit PCM
    audio = (audio * 32767).astype(np.int16)

    # 重采样到目标采样率
    if orig_rate != target_sample_rate:
        ratio = target_sample_rate / orig_rate
        n_samples = int(len(audio) * ratio)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n_samples),
            np.arange(len(audio)),
            audio
        ).astype(np.int16)
    return audio


def audio_callback(indata, frames, time_info, status):
    global is_recording, recording, silence_duration

    # 准备VAD需要的音频格式
    try:
        audio = prepare_audio_for_vad(indata, device_sample_rate)
        audio_bytes = audio.tobytes()

        # 确保音频长度符合VAD要求
        if len(audio_bytes) != 2 * frame_size:  # 16-bit = 2字节/采样
            return

        # VAD检测
        is_speech = vad.is_speech(audio_bytes, target_sample_rate)

        if is_speech:
            silence_duration = 0
            if not is_recording:
                print("\n检测到语音，开始录音...")
                is_recording = True
                recording = [indata.copy()]
            else:
                recording.append(indata.copy())
        elif is_recording:
            silence_duration += frames / device_sample_rate
            recording.append(indata.copy())
            if silence_duration >= silence_threshold:
                save_recording()
    except Exception as e:
        print(f"音频处理错误: {str(e)}")


def save_recording():
    global is_recording, recording
    if recording:
        try:
            audio_data = np.concatenate(recording)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_filename = f"recordings/output_{timestamp}.wav"
            write(output_filename, device_sample_rate, audio_data)
            print(f"保存录音: {output_filename} ({len(audio_data) / device_sample_rate:.1f}秒)")
        except Exception as e:
            print(f"保存失败: {str(e)}")
    is_recording = False
    recording = []


# 打开音频流
try:
    with sd.InputStream(
            device=selected_index,
            samplerate=device_sample_rate,
            channels=1,
            dtype="float32",
            blocksize=frame_size,
            callback=audio_callback
    ):
        print("等待语音输入... (按 Ctrl+C 停止)")
        while True:
            time.sleep(0.1)
except KeyboardInterrupt:
    print("\n程序退出")
    if is_recording:
        save_recording()