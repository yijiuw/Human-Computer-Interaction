'''-------开发日志-------
时间：3/31
内容：学习webrtcvad检测语音
'''

import webrtcvad

vad = webrtcvad.Vad(3)  # 3 是最激进的模式（严格检测语音）
sample_rate = 16000  # VAD 仅支持 16kHz
frame_duration = 30  # 30ms 帧（WebRTC VAD 要求）
frame_size = int(sample_rate * frame_duration / 1000)  # 480 采样点

# 模拟一个音频帧（静音）
audio_frame = b"\x00\x00" * frame_size  # 16-bit PCM（静音）

# 检测是否是语音
is_speech = vad.is_speech(audio_frame, sample_rate)
print("是语音吗？", is_speech)  # False（静音）