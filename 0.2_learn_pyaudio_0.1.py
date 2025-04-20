import pyaudio
import wave


def select_input_device(p):
    """选择输入设备并返回设备索引"""
    print("\n可用音频输入设备：")
    input_devices = []
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev['maxInputChannels'] > 0:
            default = "(默认)" if dev['index'] == p.get_default_input_device_info()['index'] else ""
            print(f"[{i}] {dev['name']} {default}")
            print(f"   输入通道: {dev['maxInputChannels']}, 采样率: {dev['defaultSampleRate']}Hz")
            input_devices.append(i)

    if not input_devices:
        raise Exception("未找到可用的输入设备")

    choice = int(input("\n请选择设备编号: "))
    if choice not in input_devices:
        raise ValueError("无效的设备选择")

    return choice


def record_audio(device_index, output_filename="output.wav", record_seconds=5):
    """录制音频"""
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    rate = 44100

    p = pyaudio.PyAudio()

    try:
        # 获取设备信息以确定最佳参数
        dev_info = p.get_device_info_by_index(device_index)
        channels = min(dev_info['maxInputChannels'], 2)  # 不超过2通道
        rate = int(dev_info['defaultSampleRate'])

        print(f"\n使用设备: {dev_info['name']}")
        print(f"参数: {channels}通道, {rate}Hz采样率")

        stream = p.open(
            format=format,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=chunk
        )

        print("开始录制... (按Ctrl+C停止)")
        frames = []

        try:
            for _ in range(0, int(rate / chunk * record_seconds)):
                data = stream.read(chunk)
                frames.append(data)
        except KeyboardInterrupt:
            print("\n用户中断录制")

        print("录制结束")

        stream.stop_stream()
        stream.close()

        # 保存文件
        wf = wave.open(output_filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        print(f"已保存到: {output_filename}")

    finally:
        p.terminate()


if __name__ == "__main__":
    p = pyaudio.PyAudio()
    try:
        device_index = select_input_device(p)
        record_audio(device_index)
    except Exception as e:
        print(f"错误: {str(e)}")
    finally:
        p.terminate()