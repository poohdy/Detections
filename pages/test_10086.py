#coding:utf-8
import streamlit as st
import cv2
import tempfile
import os
import mediapipe as mp
from process_frame_2 import ExerciseCounter
from thresholds import get_thresholds_beginner, get_thresholds_pro

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

st.title("AI 健身教练：实时+视频动作分析 💪🎬")

mode_select = st.radio("选择模式", ["摄像头实时识别", "上传视频识别"])
exercise = st.selectbox("选择动作", ["深蹲", "俯卧撑", "仰卧起坐", "引体向上", "跳绳"])
level = st.radio("选择难度", ["初学者", "专业"])

exercise_map = {"深蹲": "squat", "俯卧撑": "pushup", "仰卧起坐": "situp", "引体向上": "pullup", "跳绳":"jump_rope"}
exercise_eng = exercise_map[exercise]

thresholds = get_thresholds_beginner(exercise_eng) if level=="初学者" else get_thresholds_pro(exercise_eng)
counter = ExerciseCounter(exercise_eng, thresholds)
st.write(f"当前选择: {exercise} - {level}")

# ---------------- 实时摄像头 ----------------
if mode_select == "摄像头实时识别":
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                image, _ = counter.process(landmarks, image)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            stframe.image(image, channels="BGR")
    cap.release()

# ---------------- 上传视频 ----------------
elif mode_select == "上传视频识别":
    uploaded_file = st.file_uploader("上传视频文件", type=["mp4","avi","mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        stframe = st.empty()
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    image, _ = counter.process(landmarks, image)
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                out.write(image)
                stframe.image(image, channels="BGR")
        cap.release()
        out.release()

        with open(out_path, "rb") as f:
            st.download_button(
                label="下载分析后视频",
                data=f,
                file_name=f"analyzed_{uploaded_file.name}",
                mime="video/mp4"
            )
        os.remove(video_path)
        os.remove(out_path)
