'''-------开发日志-------
时间：3/31
内容：学习sounddevice的实时音频流处理
'''
#注意此代码没实现因为linux检测键盘要root权限
#如果要测试回到0.6_learn_sd_Visualize_sound.py去学习
import sounddevice as sd  # 音频输入/输出
from scipy.io.wavfile import write  # 保存 WAV 文件
import numpy as np  # 处理音频数据

# 列出所有音频设备

input_devices = [
    (i, dev["name"])
    for i, dev in enumerate(sd.query_devices())
    if dev["max_input_channels"] > 0
]

#列出设备
print("可用的输入设备：")
for i, name in input_devices:
    print(f"[{i}] {name}")


# 2. 选择设备
selected_index = int(input("输入设备编号: "))
device_info = sd.query_devices(selected_index)
sample_rate = int(device_info["default_samplerate"])  #获取设备的采样率（Hz）
output_filename = "output.wav" # 输出文件名


# 3. 实时音频流处理
recording = []  # 存储录音数据
is_recording = False  # 控制录音状态


def audio_callback(indata, frames, time_info, status):
    global is_recording, recording

    # 计算实时音量（RMS）
    volume = np.sqrt(np.mean(indata ** 2)) * 100
    #print(f"\r当前音量: {'|' * int(volume)} ({volume:.1f})", end="", flush=True)

    # 如果正在录音，保存数据
    if is_recording:
        recording.append(indata.copy())
        print()


''' 单次录制
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
'''

# 4. 打开音频流
try :
    with sd.InputStream(
        device=selected_index,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        callback=audio_callback

    ):
        print("\n开始监听麦克风...（按空格键开始/停止录音，按Ctrl+C退出）")
        while True:
            key = input()  # 等待用户按键
            if key == ' ':
                is_recording = not is_recording
                if is_recording:
                    recording.clear()  # 开始新的录音
                    print("\n开始录音...")
                else:
                    print("\n停止录音")
                    # 保存录音
                    if recording:
                        audio_data = np.concatenate(recording)
                        write(output_filename, sample_rate, audio_data)
                        print(f"音频已保存为: {output_filename}")
except KeyboardInterrupt:
    print("\n程序退出")
