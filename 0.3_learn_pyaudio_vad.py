import pyaudio
import wave
import webrtcvad  # 语音活动检测库
import numpy as np
from collections import deque
import time

# 参数配置
CHUNK = 480  # 每次读取的音频帧数（必须是10/20/30ms的帧，对应480/960/1440）
FORMAT = pyaudio.paInt16  # 16位采样格式
CHANNELS = 1  # 单声道
RATE = 16000  # 采样率（VAD通常需要16kHz）
SILENCE_TIMEOUT = 1.5  # 静音超时（秒）后停止录制
VAD_AGGRESSIVENESS = 3  # VAD检测激进程度（1-3，3最严格）
MIN_SPEECH_DURATION = 0.5  # 最小有效语音时长（秒）


class SpeechRecorder:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.audio_buffer = deque(maxlen=int(RATE * 5 / CHUNK))  # 最多缓存5秒音频
        self.is_recording = False
        self.last_voice_time = 0
        self.frames = []

    def start(self):
        """打开音频流并开始检测"""
        print(f"初始化完成，等待语音输入...（VAD模式：{VAD_AGGRESSIVENESS}）")
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=self._callback
        )
        self.stream.start_stream()

    def _callback(self, in_data, frame_count, time_info, status):
        """音频流回调函数（自动触发）"""
        # 将音频数据转为numpy数组（16bit转-32768~32767）
        audio_data = np.frombuffer(in_data, dtype=np.int16)

        # VAD检测（需要16kHz, 16bit, 单声道）
        is_speech = self.vad.is_speech(audio_data.tobytes(), RATE)

        # 语音活动处理逻辑
        if is_speech:
            self.last_voice_time = time.time()
            if not self.is_recording:
                print("\n检测到语音，开始录制...")
                self.is_recording = True
                self._save_buffer()  # 保存检测前的缓冲

        # 如果正在录制但静音超时
        elif self.is_recording and (time.time() - self.last_voice_time > SILENCE_TIMEOUT):
            print(f"\n静音超时，停止录制（有效时长：{len(self.frames) * CHUNK / RATE:.1f}s）")
            self._save_recording()
            self.is_recording = False

        # 缓冲当前数据（无论是否在录制状态）
        self.audio_buffer.append(in_data)

        return (None, pyaudio.paContinue)

    def _save_buffer(self):
        """将缓冲区的数据加入录制队列"""
        self.frames.extend(self.audio_buffer)
        self.audio_buffer.clear()

    def _save_recording(self):
        """保存有效语音片段"""
        if len(self.frames) < MIN_SPEECH_DURATION * RATE / CHUNK:
            print(f"语音过短（<{MIN_SPEECH_DURATION}s），已丢弃")
            self.frames = []
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"speech_{timestamp}.wav"

        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        print(f"已保存语音片段：{filename}")
        self.frames = []


if __name__ == "__main__":
    try:
        recorder = SpeechRecorder()
        recorder.start()

        # 保持主线程运行
        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n用户终止程序")
    finally:
        if 'recorder' in locals():
            recorder.stream.stop_stream()
            recorder.stream.close()
        pyaudio.PyAudio().terminate()