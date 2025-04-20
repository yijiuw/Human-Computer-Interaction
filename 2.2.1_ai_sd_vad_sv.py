"""-------开发日志-------
时间：4/2
内容：在保存录音时进行识别（简化版）
"""
import os
import time
import numpy as np
from collections import deque
from funasr import AutoModel
from scipy.io.wavfile import write
import sounddevice as sd
import webrtcvad


class VoiceRecorder:
    def __init__(self):
        # 初始化VAD
        self.vad = webrtcvad.Vad(3)
        self.target_sample_rate = 16000
        self.frame_duration = 30
        self.frame_size = int(self.target_sample_rate * self.frame_duration / 1000)

        # 录音参数
        self.recording = []
        self.is_recording = False
        self.silence_threshold = 1.5
        self.silence_duration = 0
        self.buffer = deque(maxlen=50)
        self.device_sample_rate = None

        # 初始化ASR模型
        self.asr_model = AutoModel(
            model="/home/he/PycharmProjects/PythonProject/model/SenseVoiceSmall",
            vad_model="fsmn-vad",
            device="cuda:0",
            trust_remote_code=True
        )

        # 创建输出目录
        self.output_dir = "recordings"
        os.makedirs(self.output_dir, exist_ok=True)

    def selected_devices(self):
        """设备选择（保持你的原有实现）"""
        input_devices = [
            (i, dev["name"])
            for i, dev in enumerate(sd.query_devices())
            if dev["max_input_channels"] > 0
        ]
        print("可用的输入设备：")
        for i, name in input_devices:
            print(f"[{i}] {name}")

        selected_index = int(input("输入设备编号: "))
        device_info = sd.query_devices(selected_index)
        return selected_index, int(device_info["default_samplerate"])

    def resample_audio(self, audio, orig_rate, target_rate):
        """音频重采样（保持你的原有实现）"""
        ratio = target_rate / orig_rate
        num_samples = int(len(audio) * ratio)
        return np.interp(
            np.linspace(0, len(audio) - 1, num_samples),
            np.arange(len(audio)),
            audio
        ).astype(np.int16)

    def audio_callback(self, indata, frames, time_info, status):
        """音频回调（稍作修改）"""
        audio = (indata[:, 0] * 32767).astype(np.int16)
        self.buffer.append(audio)

        # VAD检测
        audio_16k = self.resample_audio(audio, self.device_sample_rate, self.target_sample_rate)
        is_speech = self.vad.is_speech(audio_16k.tobytes(), self.target_sample_rate)

        if is_speech:
            self.silence_duration = 0
            if not self.is_recording:
                print("\n检测到语音，开始录音...")
                self.is_recording = True
                self.recording = list(self.buffer)[:]  # 复制缓冲区数据
            else:
                self.recording.append(audio.copy())
        elif self.is_recording:
            self.silence_duration += frames / self.device_sample_rate
            self.recording.append(audio.copy())
            if self.silence_duration >= self.silence_threshold:
                self.save_recording()

    def save_recording(self):
        """保存并识别录音（新增ASR识别）"""
        if not self.recording:
            return

        # 合并音频数据（保持int16格式）
        audio_data = np.concatenate(self.recording)

        # 保存原始录音
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"{self.output_dir}/recording_{timestamp}.wav"
        write(output_file, self.device_sample_rate, audio_data)
        print(f"\n保存录音: {output_file} ({len(audio_data) / self.device_sample_rate:.1f}秒)")

        # 执行ASR识别（转换为float32）
        try:
            audio_float32 = audio_data.astype(np.float32) / 32768.0
            res = self.asr_model.generate(
                input=audio_float32,
                language="zh",
                batch_size_s=5,
                use_itn=True
            )
            if res and res[0]['text']:
                text = res[0]['text'].split(">")[-1].strip()
                print(f"识别结果: {text}")
        except Exception as e:
            print(f"识别失败: {str(e)}")

        # 重置状态
        self.is_recording = False
        self.recording = []
        self.silence_duration = 0

    def start(self):
        """启动录音（保持你的原有实现）"""
        try:
            device_id, self.device_sample_rate = self.selected_devices()
            with sd.InputStream(
                    device=device_id,
                    samplerate=self.device_sample_rate,
                    channels=1,
                    dtype='float32',
                    blocksize=self.frame_size,
                    callback=self.audio_callback
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