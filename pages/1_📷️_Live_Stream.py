import av
import os
import sys
import streamlit as st
from streamlit_webrtc import VideoHTMLAttributes, webrtc_streamer
from aiortc.contrib.media import MediaRecorder

# 设置项目根目录，便于导入本地模块
BASE_DIR = os.path.abspath(os.path.join(__file__, '../../'))
sys.path.append(BASE_DIR)

# 导入自定义工具和分析类
from utils import get_mediapipe_pose           # 获取MediaPipe姿态模型，返回pose对象
from process_frame import ProcessFrame         # 处理每一帧，分析动作，返回标注后帧
from thresholds import get_thresholds_beginner, get_thresholds_pro  # 获取不同模式下的判定阈值

# -------------------- UI部分 --------------------
st.title('AI Fitness Trainer: Squats Analysis')  # 页面标题

# 选择模式（Beginner/Pro），决定阈值
mode = st.radio('Select Mode', ['Beginner', 'Pro'], horizontal=True)

thresholds = None  # 阈值参数字典

# 根据模式选择阈值，调用thresholds.py中的函数
if mode == 'Beginner':
    thresholds = get_thresholds_beginner()    # 返回dict，包含各关节角度等判定参数
elif mode == 'Pro':
    thresholds = get_thresholds_pro()

# 初始化动作分析主类，传入阈值和镜像参数
live_process_frame = ProcessFrame(thresholds=thresholds, flip_frame=True)
# 初始化MediaPipe姿态模型
pose = get_mediapipe_pose()

# -------------------- 下载状态管理 --------------------
if 'download' not in st.session_state:
    st.session_state['download'] = False

output_video_file = f'output_live.flv'  # 输出视频文件名

# -------------------- 视频帧处理回调 --------------------
def video_frame_callback(frame: av.VideoFrame):
    """
    输入：av.VideoFrame对象（摄像头采集的原始帧）
    输出：处理后的视频帧（带标注）
    过程：解码为ndarray，送入ProcessFrame.process分析，返回新帧
    """
    frame = frame.to_ndarray(format="rgb24")  # 解码为RGB帧
    frame, _ = live_process_frame.process(frame, pose)  # 分析动作，返回标注帧
    return av.VideoFrame.from_ndarray(frame, format="rgb24")  # 编码回av.VideoFrame

# -------------------- 录制器工厂 --------------------
def out_recorder_factory() -> MediaRecorder:
    """
    输出：MediaRecorder对象
    用于WebRTC流的录制，保存为output_video_file
    """
    return MediaRecorder(output_video_file)

# -------------------- WebRTC流设置 --------------------
ctx = webrtc_streamer(
    key="Squats-pose-analysis",                        # 唯一标识
    video_frame_callback=video_frame_callback,         # 帧处理回调
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},  # STUN服务器配置
    media_stream_constraints={"video": {"width": {'min':480, 'ideal':480}}, "audio": False},  # 视频参数
    video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, muted=False), # 播放器属性
    out_recorder_factory=out_recorder_factory          # 录制工厂
)

# -------------------- 视频下载功能 --------------------
download_button = st.empty()  # 预留下载按钮位置

if os.path.exists(output_video_file):
    with open(output_video_file, 'rb') as op_vid:
        # 提供下载按钮，点击后可下载录制视频
        download = download_button.download_button('Download Video', data=op_vid, file_name='output_live.flv')
        if download:
            st.session_state['download'] = True

# 下载后自动删除本地视频文件，避免重复下载
if os.path.exists(output_video_file) and st.session_state['download']:
    os.remove(output_video_file)
    st.session_state['download'] = False
    download_button.empty()


