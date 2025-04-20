"""-------开发日志-------
时间：4/8
内容：在3.0_sd_sv_edgeTTs.py的基础上对语言识别进行改进
目标：1.增加了语言检测机制，对VAD结果进行平滑处理，避免瞬时波动
问题：好像没解决每次录音结束后多录出来的那一部分
"""

import sounddevice as sd  # 音频输入/输出
import webrtcvad
from scipy.io.wavfile import write  # 保存 WAV 文件
import numpy as np  # 处理音频数据
from collections import deque #语言缓存区
from funasr import AutoModel
import os
import  time
import edge_tts
import cv2  #视频处理
import asyncio
from playsound import playsound

class VoiceRecorder:
    def __init__(self):
        #初始化VAD
        self.vad = webrtcvad.Vad(3) # 3 是最激进的模式（严格检测语音）
        self.vad_history = deque(maxlen=20)  # 保存最近5次VAD结果
        self.speech_ratio_threshold = 0.7  # 70%的帧为语音才判定为有效
        self.target_sample_rate = 16000 # VAD 仅支持 16kHz
        self.frame_duration = 30   # 30ms 帧（WebRTC VAD 要求）
        self.frame_size = int(self.target_sample_rate * self.frame_duration / 1000) # 480 采样点

        #录音参数设置
        self.recording = []  #保存录音数据
        self.is_recording = False #是否开启录音
        self.silence_threshold = 1.5 #静音多少秒后停止录音
        self.silence_duration = 0 #静音时间
        self.buffer = deque(maxlen=50) #录音缓存区 用于保存前0.5秒的录音
        self.device_sample_rate = None #设备采样率

        #edgeTTs参数
        self.tts_voice = "zh-CN-YunxiNeural"  #tts的音色

        # 初始化ASR模型
        self.asr_model = AutoModel(
            model="/home/he/PycharmProjects/PythonProject/model/SenseVoiceSmall",
            trust_remote_code=True,  # 允许加载远程代码（安全警告：需确保代码来源可信）
            remote_code="./model.py",  # 自定义模型代码路径
            vad_model="fsmn-vad",  # 使用FSMN语音活动检测模型
            vad_kwargs={"max_single_segment_time": 30000},  # VAD参数：设置单段语音最大时长30秒
            device="cuda:0",  # 使用GPU加速（CUDA设备0）
        )

        #保存录音参数设置
        self.output_dir = "recordings"
        os.makedirs(self.output_dir,exist_ok=True)


    @staticmethod
    def selected_devices():
        # 选择音频设备
        return_devices = []
        input_devices = [
            (i, dev["name"])
            for i, dev in enumerate(sd.query_devices())
            if dev["max_input_channels"] > 0
        ]
        print("可用的输入设备：")
        for i, name in input_devices:
            print(f"[{i}] {name}")

        selected_index = 5
        device_info = sd.query_devices(selected_index)
        device_sample_rate = int(device_info["default_samplerate"])  # 获取设备的采样率（Hz）

        return  selected_index, device_sample_rate


    @staticmethod
    def resample_audio(audio, orig_rate, target_rate):
        """将音频重采样到目标频率（用于VAD）"""
        ratio = target_rate / orig_rate
        num_samples = int(len(audio) * ratio)
        return np.interp(
            np.linspace(0, len(audio) - 1, num_samples),
            np.arange(len(audio)),
            audio
        ).astype(np.int16)

    def text_to_speech(self, text):
        """同步调用TTS生成并播放语音（使用playsound）"""
        #保存的目录和位置
        path = f'voice/output_{time.time()}.mp3'

        try:
            # 生成语音文件（同步阻塞直到完成）
            asyncio.run(edge_tts.Communicate(text, self.tts_voice).save(path))

            # 用 playsound 播放
            playsound(path)

            # 删除临时文件
            #os.remove(self.temp_audio_path)

        except Exception as e:
            print(f"语音合成/播放失败: {str(e)}")

    def audio_callback(self,indata, frames, time_info, status):
        # 转换为16-bit PCM并取单声道
        audio = (indata[:, 0] * 32767).astype(np.int16)
        self.buffer.append(audio) #缓存数据

        # 重采样到16kHz供VAD使用
        audio_16k = self.resample_audio(audio, self.device_sample_rate, self.target_sample_rate)

        # VAD 检测
        is_speech = self.vad.is_speech(audio_16k.tobytes(), self.target_sample_rate)
        self.vad_history.append(is_speech)
        smoothed_speech = sum(self.vad_history) / len(self.vad_history) > self.speech_ratio_threshold
        #print(smoothed_speech)

        #处理语言数据并保存到self.recording里
        if smoothed_speech:
            self.silence_duration = 0
            if not self.is_recording:
                print("\n检测到语音，开始录音...")
                self.is_recording = True
                self.recording.clear()
                self.recording.extend(list(self.buffer))  #注意原始self.buffer不能直接使用因为不是一个普通列表
            else:
                self.recording.append(audio.copy())  # 存储处理后的 16kHz 音频
        elif self.is_recording:
            self.silence_duration += frames / self.device_sample_rate
            self.recording.append(audio.copy())
            if self.silence_duration >= self.silence_threshold:
                self.save_recording()


    def save_recording(self):
        """保存语言数据到文件夹里并识别"""
        if self.recording:
            audio_data = np.concatenate(self.recording)

            # 保存原始录音
            output_filename = f"test/output_{int(time.time())}.wav"
            write(output_filename, int(self.device_sample_rate), audio_data.astype(np.int16))
            print(f"保存录音: {output_filename} ({len(audio_data) / self.device_sample_rate:.1f} 秒)")

            # 执行ASR识别（转换为float32）
            try:
                #audio_float32 = audio_data.astype(np.float32) / 32768.0
                res = self.asr_model.generate(
                    input=output_filename,
                    cache={},  # 缓存字典（用于增量识别）
                    language="zh",
                    batch_size_s=60, # 批处理大小（秒数）
                    use_itn=False # 启用逆文本归一化（将"123"转为"一百二十三"等）
                )
                if res and res[0]['text']:
                    text = res[0]['text'].split(">")[-1].strip()
                    print(f"识别结果: {text}")

                    #调用tts播放语音
                    #self.text_to_speech(text) #同步调用的

            except Exception as e:
                print(f"识别失败: {str(e)}")

        #重置录音参数
        self.is_recording = False
        self.recording = []
        self.silence_duration = 0



    def start(self,device_id=None):
        try:
            input_devices = self.selected_devices()
            device_id, self.device_sample_rate = input_devices
            if not input_devices:
                raise RuntimeError("未找到输入设备")

            #配置音频流
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
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        print("!")
        cv2.imshow("Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()