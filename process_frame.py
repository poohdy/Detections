import time
import cv2
import numpy as np
from utils import find_angle, get_landmark_features, draw_text, draw_dotted_line

# ProcessFrame类：用于分析每一帧的深蹲动作，计数、反馈、标注视频帧
class ProcessFrame:
    def __init__(self, thresholds, flip_frame = False):
        """
        thresholds: 判定阈值（dict），由thresholds.py生成
        flip_frame: 是否镜像画面（bool），摄像头方向适配
        初始化字体、颜色、关键点映射、状态追踪器
        """
        self.flip_frame = flip_frame
        self.thresholds = thresholds
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.linetype = cv2.LINE_AA
        self.radius = 20

        # 颜色定义（BGR格式）
        self.COLORS = {
            'blue'       : (0, 127, 255),
            'red'        : (255, 50, 50),
            'green'      : (0, 255, 127),
            'light_green': (100, 233, 127),
            'yellow'     : (255, 255, 0),
            'magenta'    : (255, 0, 255),
            'white'      : (255,255,255),
            'cyan'       : (0, 255, 255),
            'light_blue' : (102, 204, 255)
        }

        # 关键点编号映射（MediaPipe标准）
        self.dict_features = {}
        self.left_features = {
            'shoulder': 11,
            'elbow'   : 13,
            'wrist'   : 15,                    
            'hip'     : 23,
            'knee'    : 25,
            'ankle'   : 27,
            'foot'    : 31
        }
        self.right_features = {
            'shoulder': 12,
            'elbow'   : 14,
            'wrist'   : 16,
            'hip'     : 24,
            'knee'    : 26,
            'ankle'   : 28,
            'foot'    : 32
        }
        self.dict_features['left'] = self.left_features
        self.dict_features['right'] = self.right_features
        self.dict_features['nose'] = 0

        # 状态追踪器：用于计数、反馈、动作序列等
        self.state_tracker = { 
            'state_seq': [],  # 动作状态序列
            'start_inactive_time': time.perf_counter(),  # 侧面静止起始时间
            'start_inactive_time_front': time.perf_counter(),  # 正面静止起始时间
            'INACTIVE_TIME': 0.0,  # 侧面静止累计时长
            'INACTIVE_TIME_FRONT': 0.0,  # 正面静止累计时长

            # 0:后仰 1:前倾 2:膝盖超脚尖 3:深蹲过深
            'DISPLAY_TEXT' : np.full((4,), False),  # 反馈文本显示控制
            'COUNT_FRAMES' : np.zeros((4,), dtype=np.int64),  # 反馈文本显示帧数
            
            'LOWER_HIPS': False,  # 是否提示降低臀部

            'INCORRECT_POSTURE': False,  # 是否错误动作
            'prev_state': None,  # 上一帧动作状态
            
            'curr_state':None,   # 当前帧动作状态

            'SQUAT_COUNT': 0,    # 正确动作计数
            'IMPROPER_SQUAT':0   # 错误动作计数
        }

        # 反馈文本内容、位置、颜色
        self.FEEDBACK_ID_MAP = {
            0: ('BEND BACKWARDS', 215, (0, 153, 255)),
            1: ('BEND FORWARD', 215, (0, 153, 255)),
            2: ('KNEE FALLING OVER TOE', 170, (255, 80, 80)),
            3: ('SQUAT TOO DEEP', 125, (255, 80, 80))
        }

    def _get_state(self, knee_angle):
        """
        输入：膝关节角度
        输出：当前深蹲状态（'s1'/'s2'/'s3'/None）
        用于判断动作流程
        """
        knee = None        
        if self.thresholds['HIP_KNEE_VERT']['NORMAL'][0] <= knee_angle <= self.thresholds['HIP_KNEE_VERT']['NORMAL'][1]:
            knee = 1
        elif self.thresholds['HIP_KNEE_VERT']['TRANS'][0] <= knee_angle <= self.thresholds['HIP_KNEE_VERT']['TRANS'][1]:
            knee = 2
        elif self.thresholds['HIP_KNEE_VERT']['PASS'][0] <= knee_angle <= self.thresholds['HIP_KNEE_VERT']['PASS'][1]:
            knee = 3
        return f's{knee}' if knee else None

    def _update_state_sequence(self, state):
        """
        输入：当前状态
        输出：无
        维护深蹲状态序列，用于判断一次完整动作
        """
        if state == 's2':
            if (('s3' not in self.state_tracker['state_seq']) and (self.state_tracker['state_seq'].count('s2'))==0) or \
                    (('s3' in self.state_tracker['state_seq']) and (self.state_tracker['state_seq'].count('s2')==1)):
                self.state_tracker['state_seq'].append(state)
        elif state == 's3':
            if (state not in self.state_tracker['state_seq']) and 's2' in self.state_tracker['state_seq']: 
                self.state_tracker['state_seq'].append(state)

    def _show_feedback(self, frame, c_frame, dict_maps, lower_hips_disp):
        """
        输入：
            frame: 当前帧
            c_frame: 反馈文本显示帧数
            dict_maps: 反馈文本内容、位置、颜色
            lower_hips_disp: 是否显示“降低臀部”提示
        输出：标注后的frame
        功能：在画面上绘制所有反馈文本
        """
        if lower_hips_disp:
            draw_text(
                frame, 
                'LOWER YOUR HIPS', 
                pos=(30, 80),
                text_color=(0, 0, 0),
                font_scale=0.6,
                text_color_bg=(255, 255, 0)
            )  
        for idx in np.where(c_frame)[0]:
            draw_text(
                frame, 
                dict_maps[idx][0], 
                pos=(30, dict_maps[idx][1]),
                text_color=(255, 255, 230),
                font_scale=0.6,
                text_color_bg=dict_maps[idx][2]
            )
        return frame

    def process(self, frame: np.array, pose):
        """
        主入口
        输入：
            frame: 当前视频帧（np.array）
            pose: mediapipe姿态模型
        输出：
            frame: 标注后的视频帧
            play_sound: 计数/错误/重置等提示音标识
        功能：
            1. 姿态识别，提取关键点
            2. 判断摄像头对齐与否，未对齐时重置计数并提示
            3. 计算各关节角度，判断当前动作状态
            4. 更新状态序列，计数正确/错误
            5. 生成反馈文本，绘制在画面上
            6. 返回处理后帧和提示音标识
        """
        play_sound = None
        frame_height, frame_width, _ = frame.shape
        keypoints = pose.process(frame)

        if keypoints.pose_landmarks:
            ps_lm = keypoints.pose_landmarks
            # 获取各关键点坐标
            nose_coord = get_landmark_features(ps_lm.landmark, self.dict_features, 'nose', frame_width, frame_height)
            left_shldr_coord, left_elbow_coord, left_wrist_coord, left_hip_coord, left_knee_coord, left_ankle_coord, left_foot_coord = \
                get_landmark_features(ps_lm.landmark, self.dict_features, 'left', frame_width, frame_height)
            right_shldr_coord, right_elbow_coord, right_wrist_coord, right_hip_coord, right_knee_coord, right_ankle_coord, right_foot_coord = \
                get_landmark_features(ps_lm.landmark, self.dict_features, 'right', frame_width, frame_height)

            # 判断摄像头是否对齐（偏移角度过大则提示）
            offset_angle = find_angle(left_shldr_coord, right_shldr_coord, nose_coord)

            if offset_angle > self.thresholds['OFFSET_THRESH']:
                # 摄像头未对齐，显示提示并重置计数
                display_inactivity = False
                end_time = time.perf_counter()
                self.state_tracker['INACTIVE_TIME_FRONT'] += end_time - self.state_tracker['start_inactive_time_front']
                self.state_tracker['start_inactive_time_front'] = end_time

                if self.state_tracker['INACTIVE_TIME_FRONT'] >= self.thresholds['INACTIVE_THRESH']:
                    self.state_tracker['SQUAT_COUNT'] = 0
                    self.state_tracker['IMPROPER_SQUAT'] = 0
                    display_inactivity = True

                # 画出关键点
                cv2.circle(frame, nose_coord, 7, self.COLORS['white'], -1)
                cv2.circle(frame, left_shldr_coord, 7, self.COLORS['yellow'], -1)
                cv2.circle(frame, right_shldr_coord, 7, self.COLORS['magenta'], -1)

                if self.flip_frame:
                    frame = cv2.flip(frame, 1)

                if display_inactivity:
                    play_sound = 'reset_counters'
                    self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                    self.state_tracker['start_inactive_time_front'] = time.perf_counter()

                # 显示计数和提示
                draw_text(
                    frame, 
                    "CORRECT: " + str(self.state_tracker['SQUAT_COUNT']), 
                    pos=(int(frame_width*0.68), 30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(18, 185, 0)
                )  
                draw_text(
                    frame, 
                    "INCORRECT: " + str(self.state_tracker['IMPROPER_SQUAT']), 
                    pos=(int(frame_width*0.68), 80),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(221, 0, 0),
                )  
                draw_text(
                    frame, 
                    'CAMERA NOT ALIGNED PROPERLY!!!', 
                    pos=(30, frame_height-60),
                    text_color=(255, 255, 230),
                    font_scale=0.65,
                    text_color_bg=(255, 153, 0),
                ) 
                draw_text(
                    frame, 
                    'OFFSET ANGLE: '+str(offset_angle), 
                    pos=(30, frame_height-30),
                    text_color=(255, 255, 230),
                    font_scale=0.65,
                    text_color_bg=(255, 153, 0),
                ) 
                # 重置侧面静止计时
                self.state_tracker['start_inactive_time'] = time.perf_counter()
                self.state_tracker['INACTIVE_TIME'] = 0.0
                self.state_tracker['prev_state'] =  None
                self.state_tracker['curr_state'] = None
            
            # 摄像头对齐，正常分析动作
            else:
                self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                self.state_tracker['start_inactive_time_front'] = time.perf_counter()

                # 判断左右侧，选取一侧作为主分析对象
                dist_l_sh_hip = abs(left_foot_coord[1]- left_shldr_coord[1])
                dist_r_sh_hip = abs(right_foot_coord[1] - right_shldr_coord)[1]

                if dist_l_sh_hip > dist_r_sh_hip:
                    shldr_coord = left_shldr_coord
                    elbow_coord = left_elbow_coord
                    wrist_coord = left_wrist_coord
                    hip_coord = left_hip_coord
                    knee_coord = left_knee_coord
                    ankle_coord = left_ankle_coord
                    foot_coord = left_foot_coord
                    multiplier = -1
                else:
                    shldr_coord = right_shldr_coord
                    elbow_coord = right_elbow_coord
                    wrist_coord = right_wrist_coord
                    hip_coord = right_hip_coord
                    knee_coord = right_knee_coord
                    ankle_coord = right_ankle_coord
                    foot_coord = right_foot_coord
                    multiplier = 1       #左右侧镜像时为1或-1，保证弧线方向正确

                # 计算各关节与垂直方向的夹角【shldr_coord（肩）----以hip_coord（髋）为竖直参考点】
                hip_vertical_angle = find_angle(shldr_coord, np.array([hip_coord[0], 0]), hip_coord)
                # 在hip_coord（髋）画弧线表示角度
                cv2.ellipse(frame, hip_coord, (30, 30), 
                            angle = 0, startAngle = -90, endAngle = -90+multiplier*hip_vertical_angle, 
                            color = self.COLORS['white'], thickness = 3, lineType = self.linetype)
                #画虚线
                draw_dotted_line(frame, hip_coord, start=hip_coord[1]-80, end=hip_coord[1]+20, line_color=self.COLORS['blue'])


                knee_vertical_angle = find_angle(hip_coord, np.array([knee_coord[0], 0]), knee_coord)
                cv2.ellipse(frame, knee_coord, (20, 20), 
                            angle = 0, startAngle = -90, endAngle = -90-multiplier*knee_vertical_angle, 
                            color = self.COLORS['white'], thickness = 3,  lineType = self.linetype)
                
                draw_dotted_line(frame, knee_coord, start=knee_coord[1]-50, end=knee_coord[1]+20, line_color=self.COLORS['blue'])


                ankle_vertical_angle = find_angle(knee_coord, np.array([ankle_coord[0], 0]), ankle_coord)
                cv2.ellipse(frame, ankle_coord, (30, 30),
                            angle = 0, startAngle = -90, endAngle = -90 + multiplier*ankle_vertical_angle,
                            color = self.COLORS['white'], thickness = 3,  lineType=self.linetype)
                draw_dotted_line(frame, ankle_coord, start=ankle_coord[1]-50, end=ankle_coord[1]+20, line_color=self.COLORS['blue'])

                # 连线显示骨骼结构
                cv2.line(frame, shldr_coord, elbow_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, wrist_coord, elbow_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, shldr_coord, hip_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, knee_coord, hip_coord, self.COLORS['light_blue'], 4,  lineType=self.linetype)
                cv2.line(frame, ankle_coord, knee_coord,self.COLORS['light_blue'], 4,  lineType=self.linetype)
                cv2.line(frame, ankle_coord, foot_coord, self.COLORS['light_blue'], 4,  lineType=self.linetype)
                # 画出关键点
                cv2.circle(frame, shldr_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, elbow_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, wrist_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, hip_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, knee_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, ankle_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, foot_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)

                # 动作状态判定
                current_state = self._get_state(int(knee_vertical_angle))
                self.state_tracker['curr_state'] = current_state
                self._update_state_sequence(current_state)

                # 计数逻辑：判断动作流程是否完整，计数正确/错误
                if current_state == 's1':
                    if len(self.state_tracker['state_seq']) == 3 and not self.state_tracker['INCORRECT_POSTURE']:
                        self.state_tracker['SQUAT_COUNT']+=1
                        play_sound = str(self.state_tracker['SQUAT_COUNT'])
                    elif 's2' in self.state_tracker['state_seq'] and len(self.state_tracker['state_seq'])==1:
                        self.state_tracker['IMPROPER_SQUAT']+=1
                        play_sound = 'incorrect'
                    elif self.state_tracker['INCORRECT_POSTURE']:
                        self.state_tracker['IMPROPER_SQUAT']+=1
                        play_sound = 'incorrect'
                    self.state_tracker['state_seq'] = []
                    self.state_tracker['INCORRECT_POSTURE'] = False

                # 反馈逻辑：根据各关节角度判断是否需要提示
                else:
                    if hip_vertical_angle > self.thresholds['HIP_THRESH'][1]:
                        self.state_tracker['DISPLAY_TEXT'][0] = True
                    elif hip_vertical_angle < self.thresholds['HIP_THRESH'][0] and \
                         self.state_tracker['state_seq'].count('s2')==1:
                        self.state_tracker['DISPLAY_TEXT'][1] = True
                    if self.thresholds['KNEE_THRESH'][0] < knee_vertical_angle < self.thresholds['KNEE_THRESH'][1] and \
                       self.state_tracker['state_seq'].count('s2')==1:
                        self.state_tracker['LOWER_HIPS'] = True
                    elif knee_vertical_angle > self.thresholds['KNEE_THRESH'][2]:
                        self.state_tracker['DISPLAY_TEXT'][3] = True
                        self.state_tracker['INCORRECT_POSTURE'] = True
                    if (ankle_vertical_angle > self.thresholds['ANKLE_THRESH']):
                        self.state_tracker['DISPLAY_TEXT'][2] = True
                        self.state_tracker['INCORRECT_POSTURE'] = True

                # 不动计时，超时则重置计数
                display_inactivity = False
                if self.state_tracker['curr_state'] == self.state_tracker['prev_state']:
                    end_time = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME'] += end_time - self.state_tracker['start_inactive_time']
                    self.state_tracker['start_inactive_time'] = end_time
                    if self.state_tracker['INACTIVE_TIME'] >= self.thresholds['INACTIVE_THRESH']:
                        self.state_tracker['SQUAT_COUNT'] = 0
                        self.state_tracker['IMPROPER_SQUAT'] = 0
                        display_inactivity = True
                else:
                    self.state_tracker['start_inactive_time'] = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME'] = 0.0

                # 反馈文本坐标
                hip_text_coord_x = hip_coord[0] + 10
                knee_text_coord_x = knee_coord[0] + 15
                ankle_text_coord_x = ankle_coord[0] + 10

                if self.flip_frame:
                    frame = cv2.flip(frame, 1)
                    hip_text_coord_x = frame_width - hip_coord[0] + 10
                    knee_text_coord_x = frame_width - knee_coord[0] + 15
                    ankle_text_coord_x = frame_width - ankle_coord[0] + 10

                # 状态重置
                if 's3' in self.state_tracker['state_seq'] or current_state == 's1':
                    self.state_tracker['LOWER_HIPS'] = False

                self.state_tracker['COUNT_FRAMES'][self.state_tracker['DISPLAY_TEXT']]+=1

                # 显示反馈文本
                frame = self._show_feedback(frame, self.state_tracker['COUNT_FRAMES'], self.FEEDBACK_ID_MAP, self.state_tracker['LOWER_HIPS'])

                if display_inactivity:
                    play_sound = 'reset_counters'
                    self.state_tracker['start_inactive_time'] = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME'] = 0.0

                # 显示关节角度
                cv2.putText(frame, str(int(hip_vertical_angle)), (hip_text_coord_x, hip_coord[1]), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(knee_vertical_angle)), (knee_text_coord_x, knee_coord[1]+10), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(ankle_vertical_angle)), (ankle_text_coord_x, ankle_coord[1]), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)

                # 显示计数
                draw_text(
                    frame, 
                    "CORRECT: " + str(self.state_tracker['SQUAT_COUNT']), 
                    pos=(int(frame_width*0.68), 30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(18, 185, 0)
                )  
                draw_text(
                    frame, 
                    "INCORRECT: " + str(self.state_tracker['IMPROPER_SQUAT']), 
                    pos=(int(frame_width*0.68), 80),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(221, 0, 0),
                )  

                # 反馈文本显示时长控制
                self.state_tracker['DISPLAY_TEXT'][self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']] = False
                self.state_tracker['COUNT_FRAMES'][self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']] = 0    
                self.state_tracker['prev_state'] = current_state

        else:
            # 未检测到人体关键点，重置部分状态并显示计数
            if self.flip_frame:
                frame = cv2.flip(frame, 1)

            end_time = time.perf_counter()
            self.state_tracker['INACTIVE_TIME'] += end_time - self.state_tracker['start_inactive_time']

            display_inactivity = False

            if self.state_tracker['INACTIVE_TIME'] >= self.thresholds['INACTIVE_THRESH']:
                self.state_tracker['SQUAT_COUNT'] = 0
                self.state_tracker['IMPROPER_SQUAT'] = 0
                display_inactivity = True

            self.state_tracker['start_inactive_time'] = end_time

            draw_text(
                frame, 
                "CORRECT: " + str(self.state_tracker['SQUAT_COUNT']), 
                pos=(int(frame_width*0.68), 30),
                text_color=(255, 255, 230),
                font_scale=0.7,
                text_color_bg=(18, 185, 0)
            )  
            draw_text(
                frame, 
                "INCORRECT: " + str(self.state_tracker['IMPROPER_SQUAT']), 
                pos=(int(frame_width*0.68), 80),
                text_color=(255, 255, 230),
                font_scale=0.7,
                text_color_bg=(221, 0, 0),
            )  

            if display_inactivity:
                play_sound = 'reset_counters'
                self.state_tracker['start_inactive_time'] = time.perf_counter()
                self.state_tracker['INACTIVE_TIME'] = 0.0

            # 重置所有状态变量
            self.state_tracker['prev_state'] =  None
            self.state_tracker['curr_state'] = None
            self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
            self.state_tracker['INCORRECT_POSTURE'] = False
            self.state_tracker['DISPLAY_TEXT'] = np.full((5,), False)
            self.state_tracker['COUNT_FRAMES'] = np.zeros((5,), dtype=np.int64)
            self.state_tracker['start_inactive_time_front'] = time.perf_counter()

        # 返回处理后的画面和声音反馈标识
        return frame, play_sound

