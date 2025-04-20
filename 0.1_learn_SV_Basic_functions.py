'''-------开发日志-------
时间：3/25
内容：测试和学习senceoice的基本功能和用法
目标：1.对一个中文音频将其识别出来输出到屏幕上
'''

from funasr import AutoModel

# 设置模型路径（指向SenseVoiceSmall模型目录）
model_dir = "/home/he/PycharmProjects/PythonProject/model/SenseVoiceSmall"

# 初始化语音识别模型
model = AutoModel(
    model=model_dir,  # 指定模型路径
    trust_remote_code=True,  # 允许加载远程代码（安全警告：需确保代码来源可信）
    remote_code="./model.py",  # 自定义模型代码路径
    vad_model="fsmn-vad",  # 使用FSMN语音活动检测模型
    vad_kwargs={"max_single_segment_time": 30000},  # VAD参数：设置单段语音最大时长30秒
    device="cuda:0",  # 使用GPU加速（CUDA设备0）
)

# 中文语音识别示例
res = model.generate(
    input=f"/home/he/PycharmProjects/PythonProject/test/output_1743585580.wav",  # 输入音频文件路径
    cache={},  # 缓存字典（用于增量识别）
    language="auto",  # 自动检测语言（可指定"zh"/"en"/"yue"/"ja"/"ko"/"nospeech"）
    use_itn=False,  # 启用逆文本归一化（将"123"转为"一百二十三"等）
    batch_size_s=60,  # 批处理大小（秒数）
    merge_vad=True,  # 合并VAD检测的短语音片段
    merge_length_s=15,  # 合并后的最大片段长度（秒）
)

#对输出结果进行处理
'''
res 的内容是
[{
    'key': 'zh', 
    'text': '<|zh|><|NEUTRAL|><|Speech|><|woitn|>开饭时间早上九点至下午五点'
}]
所以res[0]['text']提取出
'<|zh|><|NEUTRAL|><|Speech|><|woitn|>开饭时间早上九点至下午五点'
在用.split(">")分割
[
    '<|zh', 
    '<|NEUTRAL', 
    '<|Speech', 
    '<|woitn', 
    '开饭时间早上九点至下午五点'
]
'''
print(res[0]['text'].split(">")[-1])