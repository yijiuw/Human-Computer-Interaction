import time
import cv2
import mediapipe as mp

# 初始化MediaPipe手部模型
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, #动态视频模式（设为True则适用于静态图片）
    max_num_hands=2 #最多检测2只手
)
mp_draw = mp.solutions.drawing_utils #用于绘制关键点和连线

cap = cv2.VideoCapture(0) #选择默认摄像头

pTime = 0
cTime = 0

while True:
    ret, frame = cap.read() #读取摄像头的一帧图像

    # 转为RGB并处理  MediaPipe需要RGB格式而OpenCV默认的格式是BGR格式
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #调用MediaPipe模型处理图像，返回检测结果results，包含手部关键点坐标。
    results = hands.process(rgb_frame)

    # 绘制关键点
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    #计算fps
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(frame,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,
    (255,0,255),3)

    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()