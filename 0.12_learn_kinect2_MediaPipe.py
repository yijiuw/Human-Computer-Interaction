"""-------开发日志-------
时间：4/22
内容：学习python调用kinect的话题并用于手势识别
目标：1.同上
"""
import rospy
import numpy as np
from sensor_msgs.msg import Image
import cv2
import mediapipe as mp

# 初始化 MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils




def ros_to_cv(ros_image):
    # 仅适用于未压缩的 BGR/RGB 图像
    if ros_image.encoding == 'bgr8':
        dtype = np.uint8
        channels = 3
    elif ros_image.encoding == 'mono8':
        dtype = np.uint8
        channels = 1
    else:
        raise ValueError("Unsupported encoding")

    # 转换为 numpy 数组
    cv_image = np.frombuffer(ros_image.data, dtype=dtype).copy()  #注意ros的传递过来的图像信息是只读的
    return cv_image.reshape(ros_image.height, ros_image.width, channels)


def image_callback(msg):
    try:
        # 将 ROS Image 消息转为 OpenCV 格式
        cv_image = ros_to_cv(msg)

        # 处理图像（MediaPipe）MediaPipe需要RGB格式而OpenCV默认的格式是BGR格式
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)

        # 绘制手部关键点
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(cv_image, landmarks, mp_hands.HAND_CONNECTIONS)

        # 显示结果
        cv2.imshow("Kinect + MediaPipe", cv_image)
        cv2.waitKey(1)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    # 初始化 ROS 节点
    rospy.init_node('kinect_mediapipe', anonymous=True)

    # 订阅 Kinect 的 RGB 话题（根据实际话题调整）
    rospy.Subscriber("/kinect2/hd/image_color", Image, image_callback)

    print("Waiting for Kinect data... Press Ctrl+C to exit.")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()