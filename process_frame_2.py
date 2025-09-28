#coding:utf-8
import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

class BufferList:
    def __init__(self, buffer_time, default_value=0):
        self.buffer = [default_value] * buffer_time
    def push(self, value):
        self.buffer.pop(0)
        self.buffer.append(value)
    def max(self):
        return max(self.buffer)
    def min(self):
        return min(filter(lambda x: x is not None, self.buffer), default=0)
    def smooth_update(self, old_value, new_value, alpha=0.5):
        return alpha * new_value + (1 - alpha) * old_value

class ExerciseCounter:
    def __init__(self, exercise_type, thresholds):
        self.exercise_type = exercise_type
        self.thresholds = thresholds
        self.counter = 0
        self.stage = None
        self.warning = ""

        if self.exercise_type == "jump_rope":
            self.buffers = {
                "center_y": BufferList(thresholds["buffer_time"]),
                "center_y_up": BufferList(thresholds["buffer_time"]),
                "center_y_down": BufferList(thresholds["buffer_time"]),
                "center_y_flip": BufferList(thresholds["buffer_time"]),
                "center_y_pref_flip": BufferList(thresholds["buffer_time"]),
            }
            self.cy_max = 100
            self.cy_min = 100
            self.flip_flag = thresholds["flag_high"]

    def process(self, landmarks, image):
        angle = 0
        self.warning = ""

        if self.exercise_type == "squat":
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            angle = calculate_angle(hip, knee, ankle)

            # 动作计数
            if angle > self.thresholds["max_angle"]:
                self.stage = "down"
            if angle < self.thresholds["min_angle"] and self.stage == 'down':
                self.stage = "up"
                self.counter += 1


            # 不标准动作提示（仅在动作阶段检测）
            if self.stage in ["down", "up"]:
                if angle < self.thresholds["min_angle"]:
                    self.warning = "too deep"
                elif angle > self.thresholds["max_angle"] + 10:
                    self.warning = "lower you body"

        elif self.exercise_type == "pushup":
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            angle = calculate_angle(shoulder, elbow, wrist)
            if angle > self.thresholds["max_angle"]:
                self.stage = "down"
            if angle < self.thresholds["min_angle"] and self.stage == 'down':
                self.stage = "up"
                self.counter += 1

        elif self.exercise_type == "situp":
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            angle = calculate_angle(shoulder, hip, knee)
            if angle > self.thresholds["max_angle"]:
                self.stage = "down"
            if angle < self.thresholds["min_angle"] and self.stage == 'down':
                self.stage = "up"
                self.counter += 1

        elif self.exercise_type == "pullup":
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            angle = calculate_angle(shoulder, elbow, wrist)
            if angle > self.thresholds["max_angle"]:
                self.stage = "down"
            if angle < self.thresholds["min_angle"] and self.stage == 'down':
                self.stage = "up"
                self.counter += 1

        elif self.exercise_type == "jump_rope":
            hip_indices = [23,24]
            shoulder_indices = [11,12]
            image_h, image_w, _ = image.shape
            hip_points = [(landmarks[i].x*image_w, landmarks[i].y*image_h) for i in hip_indices]
            shoulder_points = [(landmarks[i].x*image_w, landmarks[i].y*image_h) for i in shoulder_indices]
            cy = int(np.mean([p[1] for p in hip_points]))
            cy_shoulder_hip = int(np.mean([p[1] for p in hip_points])) - int(np.mean([p[1] for p in shoulder_points]))

            buf = self.buffers
            buf["center_y"].push(cy)
            self.cy_max = buf["center_y"].smooth_update(self.cy_max, buf["center_y"].max())
            buf["center_y_up"].push(self.cy_max)
            self.cy_min = buf["center_y"].smooth_update(self.cy_min, buf["center_y"].min())
            buf["center_y_down"].push(self.cy_min)
            prev_flag = self.flip_flag

            if (cy > self.cy_max - self.thresholds["up_ratio"]*(self.cy_max-self.cy_min)
                and self.flip_flag == self.thresholds["flag_low"]):
                self.flip_flag = self.thresholds["flag_high"]
            elif (cy < self.cy_min + self.thresholds["down_ratio"]*(self.cy_max-self.cy_min)
                and self.flip_flag == self.thresholds["flag_high"]):
                self.flip_flag = self.thresholds["flag_low"]

            buf["center_y_flip"].push(self.flip_flag)
            buf["center_y_pref_flip"].push(prev_flag)

            if prev_flag < self.flip_flag:
                self.counter += 1

        # 绘制信息
        if self.exercise_type != "jump_rope":
            cv2.putText(image, f"Angle: {int(angle)}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(image, f"Count: {self.counter}", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if self.stage:
            cv2.putText(image, f"Stage: {self.stage}", (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        if self.warning:
            cv2.putText(image, f"⚠ {self.warning}", (50, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return image, self.counter
