
'''-------开发日志-------
时间：3/27
内容：下面是ai生成的基于sounddevice和vad实时检测语音输入
'''

import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import webrtcvad
import os
import time
from collections import deque


class VoiceRecorder:
    def __init__(self):
        self.vad = webrtcvad.Vad(3)
        self.chunk_size = 480  # 30ms的帧
        self.buffer = deque(maxlen=50)
        self.recording = []
        self.is_recording = False
        self.silence_threshold = 1.5
        self.min_voice_duration = 0.5
        self.output_dir = "recordings"
        os.makedirs(self.output_dir, exist_ok=True)

    def list_devices(self):
        print("\n=== 可用输入设备 ===")
        devices = sd.query_devices()
        input_devices = []

        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                default = "(默认)" if i == sd.default.device[0] else ""
                print(f"[{i}] {dev['name']} {default}")
                print(f"    输入通道: {dev['max_input_channels']}, 支持采样率: {dev['default_samplerate']}Hz")
                input_devices.append((i, int(dev['default_samplerate'])))

        return input_devices

    def resample_audio(self, audio, orig_rate, target_rate):
        """将音频重采样到目标频率（用于VAD）"""
        ratio = target_rate / orig_rate
        n_samples = int(len(audio) * ratio)
        return np.interp(
            np.linspace(0, len(audio) - 1, n_samples),
            np.arange(len(audio)),
            audio
        ).astype(np.int16)

    def callback(self, indata, frames, time, status):
        # 转换为16-bit PCM并取单声道
        audio = (indata[:, 0] * 32767).astype(np.int16)

        # 重采样到16kHz供VAD使用
        audio_16k = self.resample_audio(audio, self.sample_rate, 16000)

        self.buffer.append(audio)

        # VAD检测
        is_speech = self.vad.is_speech(audio_16k.tobytes(), 16000)

        if is_speech:
            self.silence_duration = 0
            if not self.is_recording:
                print("\n检测到语音，开始录制...")
                self.is_recording = True
                self.recording.extend(list(self.buffer))
            else:
                self.recording.append(audio)
        elif self.is_recording:
            self.silence_duration += frames / self.sample_rate
            self.recording.append(audio)

            if self.silence_duration >= self.silence_threshold:
                self.save_recording()

    def save_recording(self):
        try:
            audio_data = np.concatenate(self.recording)
            duration = len(audio_data) / self.sample_rate

            if duration >= self.min_voice_duration:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"{self.output_dir}/recording_{timestamp}.wav"

                # 确保数据是int16类型
                if audio_data.dtype != np.int16:
                    audio_data = audio_data.astype(np.int16)

                # 确保采样率是整数
                write(filename, int(self.sample_rate), audio_data)
                print(f"保存语音段: {filename} ({duration:.1f}秒)")

        except Exception as e:
            print(f"保存失败: {str(e)}")
        finally:
            self.recording = []
            self.is_recording = False
            self.silence_duration = 0

    def start(self, device_id=None):
        try:
            input_devices = self.list_devices()
            print(input_devices)
            if not input_devices:
                raise RuntimeError("未找到输入设备")

            # 让用户选择设备
            for i, (dev_id, rate) in enumerate(input_devices):
                print(f"{i}: 设备ID {dev_id} (采样率 {rate}Hz)")

            choice = int(input("\n选择设备编号: "))
            device_id, self.sample_rate = input_devices[choice]

            dev_info = sd.query_devices(device_id)
            print(f"\n使用设备: [{device_id}] {dev_info['name']}")
            print(f"使用采样率: {self.sample_rate}Hz")

            # 配置音频流
            with sd.InputStream(
                    device=device_id,
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype='float32',
                    blocksize=self.chunk_size,
                    callback=self.callback
            ):
                print("\n等待语音输入... (按Ctrl+C停止)")
                while True:
                    sd.sleep(1000)

        except KeyboardInterrupt:
            print("\n停止录制")
            if self.is_recording:
                self.save_recording()
        except Exception as e:
            print(f"错误: {str(e)}")


if __name__ == "__main__":
    recorder = VoiceRecorder()
    recorder.start()