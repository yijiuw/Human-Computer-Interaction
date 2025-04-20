"""-------开发日志-------
时间：4/16
内容：识别手势并将其识别内容写入文件gesture_data.txt中
目标：1.使用类封装手势识别逻辑
问题：
"""

import cv2
import mediapipe as mp


class GestureDetector:
    def __init__(self):
        """初始化数据"""
        self.camera_index = 0  #摄像头设备索引（默认0）
        self.output_file = "gesture_data.txt"  #手势数据输出文件路径

        # 初始化MediaPipe手部模型
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,  # 视频流模式
            max_num_hands=2,  # 最多检测2只手
            min_detection_confidence=0.7  # 置信度阈值
        )
        self.mp_draw = mp.solutions.drawing_utils   #用于绘制关键点和连线

        # 摄像头对象（稍后在start方法中初始化）
        self.cap = None

    def identify_hand(self, frame):
        """检测当前帧中的手的点位"""
        # MediaPipe需要RGB格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 存储关键点列表
        hand_landmarks_list = []

        # 检测手部关键点
        results = self.hands.process(rgb_frame)

        # 绘制关键点（可选）
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                # 提取每个关键点的坐标
                hand_landmarks = []
                for landmark in landmarks.landmark:
                    # 获取关键点的x,y坐标（相对于图像宽度和高度的比例）
                    h, w, c = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    hand_landmarks.append((cx, cy))
                hand_landmarks_list.append(hand_landmarks)



        return hand_landmarks_list,frame


    def recognize_gestures(self,hand_landmarks_list):
        """识别手势"""
        gestures = []  #存储手势识别内容

        if not hand_landmarks_list:
            gestures.append("None")
            gestures.append("没有识别到手势")
            return gestures  # 没有检测到手
        #手势识别
        for landmarks in hand_landmarks_list:
            # 获取关键点索引
            # 手腕和手掌根部
            wrist = landmarks[0]  # 手腕中心点

            # 拇指 (Thumb)
            thumb_cmc = landmarks[1]  # 拇指根部（连接手掌）
            thumb_mcp = landmarks[2]  # 拇指第一关节（掌指关节）
            thumb_ip = landmarks[3]  # 拇指第二关节（指间关节）
            thumb_tip = landmarks[4]  # 拇指指尖

            # 食指 (Index Finger)
            index_mcp = landmarks[5]  # 食指根部（掌指关节）
            index_pip = landmarks[6]  # 食指第一关节（近端指间关节）
            index_dip = landmarks[7]  # 食指第二关节（远端指间关节）
            index_tip = landmarks[8]  # 食指指尖

            # 中指 (Middle Finger)
            middle_mcp = landmarks[9]  # 中指根部（掌指关节）
            middle_pip = landmarks[10]  # 中指第一关节
            middle_dip = landmarks[11]  # 中指第二关节
            middle_tip = landmarks[12]  # 中指指尖

            # 无名指 (Ring Finger)
            ring_mcp = landmarks[13]  # 无名指根部
            ring_pip = landmarks[14]  # 无名指第一关节
            ring_dip = landmarks[15]  # 无名指第二关节
            ring_tip = landmarks[16]  # 无名指指尖

            # 小指 (Pinky Finger)
            pinky_mcp = landmarks[17]  # 小指根部
            pinky_pip = landmarks[18]  # 小指第一关节
            pinky_dip = landmarks[19]  # 小指第二关节
            pinky_tip = landmarks[20]  # 小指指尖

            # 判断手指是否伸直：指尖的 y 坐标 < 第二关节的 y 坐标（因为摄像头坐标系 y 向下）
            index_straight = (index_tip[1] < index_pip[1])
            middle_straight = (middle_tip[1] < middle_pip[1])
            ring_straight = (ring_tip[1] < ring_pip[1])
            pinky_straight = (pinky_tip[1] < pinky_pip[1])
            thumb_straight = (thumb_tip[0] > thumb_ip[0])  # 拇指用 x 坐标判断（横向）

            # 数字1：只有食指伸直
            if index_straight and not (middle_straight or ring_straight or pinky_straight or thumb_straight):
                gestures.append("1")
                gestures.append("视频里的手势是数字一")
            # 数字2：食指 + 中指伸直
            elif index_straight and middle_straight and not (ring_straight or pinky_straight or thumb_straight):
                gestures.append("2")
                gestures.append("视频里的手势是数字二")
            elif index_straight and middle_straight and ring_straight and not (pinky_straight or  thumb_straight):
                gestures.append("3")
                gestures.append("视频里的手势是数字三")
            elif index_straight and middle_straight and ring_straight and  pinky_straight and not thumb_straight:
                gestures.append("4")
                gestures.append("视频里的手势是数字四")
            elif index_straight and middle_straight and ring_straight and  pinky_straight and  thumb_straight:
                gestures.append("5")
                gestures.append("视频里的手势是数字五")
            else:
                gestures.append("None")
                gestures.append("没有识别到手势")
        return gestures
    def write_gesture_to_file(self, gesture):
        """将手势结果写入文件"""
        with open(self.output_file, "w") as f:
            f.write(gesture)

    def start(self):
        """启动摄像头并开始检测手势"""
        self.cap = cv2.VideoCapture(self.camera_index)

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # 检测手势
                results,frame= self.identify_hand(frame)
                gestures = self.recognize_gestures(results)
                text = gestures[0]
                china_text = gestures[1]

                #将识别结果显示在画面里
                cv2.putText(frame,text, (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                            (255, 0, 255), 3)

                # 写入文件
                self.write_gesture_to_file(china_text)

                # 显示画面
                cv2.imshow("Gesture Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.release_resources()

    def release_resources(self):
        """释放摄像头和MediaPipe资源"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()  # 释放MediaPipe资源


if __name__ == "__main__":
    # 使用示例
    detector = GestureDetector()
    detector.start()