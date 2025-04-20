"""-------开发日志-------
时间：4/15
内容：测试opencv能不能在子程序中运行
目标：1.同上
结果：居然可以在子线程中实现，ai说的是
1. OpenCV 的 GUI 线程模型

OpenCV 的窗口系统（cv2.imshow + cv2.waitKey）对线程有限制：

    Linux/macOS：依赖 X11 或 Cocoa GUI 框架，通常要求 主线程 运行窗口。

    Windows：依赖 Win32 API，相对宽松，但仍有潜在问题。

你的第一个代码能运行，可能是因为：

    系统/OpenCV 版本对多线程窗口的支持较好（尤其是 Windows）。

    没有其他线程竞争 GUI 资源（简单场景）。

而第二个代码失败，可能是因为：

    主线程被其他任务（如音频处理）阻塞，导致 OpenCV 窗口事件无法处理。

    多线程竞争导致 OpenCV 内部状态混乱。
我表示很难理解。现在去尝试第二种结局方法就是在主线程运行 OpenCV 窗口
"""
import time
import cv2
import mediapipe as mp
import threading

def gesture_recognition():

    cap = cv2.VideoCapture(0)
    pTime = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) != 0 else 0
        pTime = cTime

        cv2.putText(frame, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        cv2.imshow("Gesture Recognition", frame)
        print("2")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 启动手势识别线程
gesture_thread = threading.Thread(target=gesture_recognition)
gesture_thread.daemon = True  # 设为守护线程，主线程结束时自动退出
gesture_thread.start()

# 主线程可以继续执行其他任务
print("主线程正在运行...")
while True:
    time.sleep(1)
    print("主线程仍在运行...")