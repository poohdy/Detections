#coding:utf-8
"""
阈值配置文件
支持动作：深蹲 / 俯卧撑 / 仰卧起坐 / 引体向上 / 跳绳
支持模式：初学者 / 专业
"""

def get_thresholds_beginner(exercise_type='squat'):
    if exercise_type == 'squat':
        return {"min_angle": 70, "max_angle": 160}
    elif exercise_type == 'pushup':
        return {"min_angle": 130, "max_angle": 160}
    elif exercise_type == 'situp':
        return {"min_angle": 40, "max_angle": 100}
    elif exercise_type == 'pullup':
        return {"min_angle": 60, "max_angle": 160}
    elif exercise_type == 'jump_rope':
        return {"buffer_time":50, "dy_ratio":0.3, "up_ratio":0.55, "down_ratio":0.35, "flag_low":150, "flag_high":250}
    else:
        raise ValueError(f"Unknown exercise type: {exercise_type}")

def get_thresholds_pro(exercise_type='squat'):
    if exercise_type == 'squat':
        return {"min_angle": 80, "max_angle": 170}
    elif exercise_type == 'pushup':
        return {"min_angle": 140, "max_angle": 165}
    elif exercise_type == 'situp':
        return {"min_angle": 50, "max_angle": 110}
    elif exercise_type == 'pullup':
        return {"min_angle": 70, "max_angle": 165}
    elif exercise_type == 'jump_rope':
        return {"buffer_time":50, "dy_ratio":0.3, "up_ratio":0.55, "down_ratio":0.35, "flag_low":150, "flag_high":250}
    else:
        raise ValueError(f"Unknown exercise type: {exercise_type}")
