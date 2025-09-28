#coding:utf-8
import cv2
import streamlit as st
import mediapipe as mp
from process_frame_2 import ExerciseCounter
from thresholds import get_thresholds_beginner, get_thresholds_pro

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# UI 界面
st.title("AI 动作识别健身教练 💪")
exercise = st.selectbox("选择动作", ["深蹲", "俯卧撑", "仰卧起坐"])
mode = st.radio("选择模式", ["初学者", "专业"])

# 中文 -> 英文映射
exercise_map = {"深蹲": "squat", "俯卧撑": "pushup", "仰卧起坐": "situp"}
exercise_eng = exercise_map[exercise]

# 加载阈值
thresholds = get_thresholds_beginner(exercise_eng) if mode == "初学者" else get_thresholds_pro(exercise_eng)

st.write(f"当前选择: {exercise} - {mode}")
stframe = st.empty()

# 打开摄像头
cap = cv2.VideoCapture(0)

counter = ExerciseCounter(exercise_eng, thresholds)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Mediapipe 处理
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            image, count = counter.process(landmarks, image)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        stframe.image(image, channels="BGR")

cap.release()
