#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/FunAudioLLM/SenseVoice). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

# 导入FunASR库中的自动模型加载类和后处理工具
# 导入FunASR库中的自动模型加载类和后处理工具
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

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
    input=f"{model.model_path}/example/zh.mp3",  # 输入音频文件路径
    cache={},  # 缓存字典（用于增量识别）
    language="auto",  # 自动检测语言（可指定"zh"/"en"/"yue"/"ja"/"ko"/"nospeech"）
    use_itn=True,  # 启用逆文本归一化（将"123"转为"一百二十三"等）
    batch_size_s=60,  # 批处理大小（秒数）
    merge_vad=True,  # 合并VAD检测的短语音片段
    merge_length_s=15,  # 合并后的最大片段长度（秒）
)

# 对识别结果进行后处理（添加标点、格式化等）
text = rich_transcription_postprocess(res[0]["text"])
# 打印识别结果
print(text)