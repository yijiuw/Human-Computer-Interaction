"""-------开发日志-------
时间：4/2
内容：ai生成的VAD语音检测 + SenseVoice实时识别
问题：是对语言内容实时识别，识别列表刚有一点点就识别了
"""
import os
import time
import queue
import threading
import numpy as np
from collections import deque
from funasr import AutoModel
import sounddevice as sd
from scipy.io.wavfile import write
import webrtcvad


class VoiceRecognitionSystem:
    def __init__(self):
        # 初始化VAD
        self.vad = webrtcvad.Vad(3)  # 激进模式
        self.vad_sample_rate = 16000  # VAD要求16kHz
        self.frame_duration = 30  # 30ms帧
        self.vad_frame_size = int(self.vad_sample_rate * self.frame_duration / 1000)

        # 录音参数
        self.device_sample_rate = None
        self.silence_threshold = 1.5  # 静音停止阈值(秒)
        self.is_recording = False
        self.silence_duration = 0
        self.recording_buffer = deque(maxlen=50)  # 环形缓冲区

        # ASR处理队列
        self.asr_queue = queue.Queue()

        # 初始化ASR模型
        self.asr_model = AutoModel(
            model="/home/he/PycharmProjects/PythonProject/model/SenseVoiceSmall",
            vad_model="fsmn-vad",
            device="cuda:0",
            trust_remote_code=True
        )

        # 创建输出目录
        os.makedirs("recordings", exist_ok=True)

    def selected_devices(self):
        return_devices = []
        input_devices = [
            (i, dev["name"])
            for i, dev in enumerate(sd.query_devices())
            if dev["max_input_channels"] > 0
        ]
        print("可用的输入设备：")
        for i, name in input_devices:
            print(f"[{i}] {name}")

        selected_index = int(input("输入设备编号: ")) #选择设备
        device_info = sd.query_devices(selected_index)
        device_sample_rate = int(device_info["default_samplerate"])  # 获取设备的采样率（Hz）
        return_devices.append((selected_index,device_sample_rate))

        return  selected_index, device_sample_rate

    def resample_for_vad(self, audio, orig_rate):
        """重采样音频供VAD使用"""
        ratio = self.vad_sample_rate / orig_rate
        n_samples = int(len(audio) * ratio)
        return np.interp(
            np.linspace(0, len(audio) - 1, n_samples),
            np.arange(len(audio)),
            audio
        ).astype(np.int16)

    def audio_callback(self, indata, frames, time_info, status):
        """音频流回调函数"""
        # 转换为单声道16-bit PCM
        audio = (indata[:, 0] * 32767).astype(np.int16)
        self.recording_buffer.append(audio)

        # VAD处理
        audio_16k = self.resample_for_vad(audio, self.device_sample_rate)
        is_speech = self.vad.is_speech(audio_16k.tobytes(), self.vad_sample_rate)

        if is_speech:
            self.silence_duration = 0
            if not self.is_recording:
                print("\n检测到语音，开始识别...")
                self.is_recording = True
                # 将缓冲区数据加入ASR队列
                for buf_audio in list(self.recording_buffer):
                    self.asr_queue.put(buf_audio.copy())
            # 持续添加新数据
            self.asr_queue.put(audio.copy())
        elif self.is_recording:
            self.silence_duration += frames / self.device_sample_rate
            self.asr_queue.put(audio.copy())
            if self.silence_duration >= self.silence_threshold:
                self.stop_recording()

    def asr_worker(self):
        """ASR识别工作线程"""
        while True:
            # 收集至少1秒的音频(16000 samples)
            audio_chunks = []
            while len(audio_chunks) < self.vad_sample_rate // self.vad_frame_size:
                try:
                    audio_chunks.append(self.asr_queue.get(timeout=1))
                except queue.Empty:
                    if not self.is_recording:
                        break

            if audio_chunks:
                audio_data = np.concatenate(audio_chunks)

                # 执行ASR识别
                try:
                    res = self.asr_model.generate(
                        input=audio_data,
                        language="zh",
                        batch_size_s=5,
                        use_itn=True
                    )
                    if res and res[0]['text']:
                        text = res[0]['text'].split(">")[-1].strip()
                        if text and text != '<|nospeech|>':
                            print(f"\r识别结果: {text}", end="", flush=True)
                except Exception as e:
                    print(f"\nASR错误: {str(e)}")

    def stop_recording(self):
        """停止录音并保存"""
        if self.is_recording:
            print("\n检测到静音，停止识别")
            self.is_recording = False

            # 保存最后一段录音
            if self.recording_buffer:
                audio_data = np.concatenate(self.recording_buffer)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_file = f"recordings/recording_{timestamp}.wav"
                write(output_file, self.device_sample_rate, audio_data)

    def run(self):
        # 列出并选择设备
        input_devices = self.selected_devices()
        device_id, self.device_sample_rate = input_devices

        # 启动ASR线程
        asr_thread = threading.Thread(target=self.asr_worker, daemon=True)
        asr_thread.start()

        try:
            with sd.InputStream(
                    device=input_devices,
                    samplerate=self.device_sample_rate,
                    channels=1,
                    dtype='float32',
                    blocksize=self.vad_frame_size,
                    callback=self.audio_callback
            ):
                print("等待语音输入... (按Ctrl+C停止)")
                while True:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n程序退出")
            self.stop_recording()


if __name__ == "__main__":
    recognizer = VoiceRecognitionSystem()
    recognizer.run()