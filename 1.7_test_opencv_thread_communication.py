"""-------开发日志-------
时间：4/15
内容：测试opencv能不能在子程序中获取，然后在主程序里显示并处理
目标：1.同上
"""
import time
import cv2
import mediapipe as mp
import threading
import queue

frame_queue = queue.Queue(maxsize=1)

def gesture_recognition():
    """摄像头子线程：只采集帧，不显示"""
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not frame_queue.full():  # 避免阻塞
            frame_queue.put(frame.copy())  # 传递帧数据
    cap.release()

# 启动手势识别线程
gesture_thread = threading.Thread(target=gesture_recognition)
gesture_thread.daemon = True  # 设为守护线程，主线程结束时自动退出
gesture_thread.start()

# 主线程可以继续执行其他任务
print("主线程正在运行...")
while True:
    if not frame_queue.empty():
        frame = frame_queue.get()
        print("1")
        cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) == 27:  # ESC键退出
        break

