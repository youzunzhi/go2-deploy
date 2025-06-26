# constants.py
# 运动API常量定义

# 机器人运动API ID
ROBOT_SPORT_API_ID_DAMP = 1001
ROBOT_SPORT_API_ID_BALANCESTAND = 1002
ROBOT_SPORT_API_ID_STOPMOVE = 1003
ROBOT_SPORT_API_ID_STANDUP = 1004
ROBOT_SPORT_API_ID_STANDDOWN = 1005
ROBOT_SPORT_API_ID_RECOVERYSTAND = 1006
ROBOT_SPORT_API_ID_EULER = 1007
ROBOT_SPORT_API_ID_MOVE = 1008
ROBOT_SPORT_API_ID_SIT = 1009
ROBOT_SPORT_API_ID_RISESIT = 1010
ROBOT_SPORT_API_ID_SWITCHGAIT = 1011
ROBOT_SPORT_API_ID_TRIGGER = 1012
ROBOT_SPORT_API_ID_BODYHEIGHT = 1013
ROBOT_SPORT_API_ID_FOOTRAISEHEIGHT = 1014
ROBOT_SPORT_API_ID_SPEEDLEVEL = 1015
ROBOT_SPORT_API_ID_HELLO = 1016
ROBOT_SPORT_API_ID_STRETCH = 1017
ROBOT_SPORT_API_ID_TRAJECTORYFOLLOW = 1018
ROBOT_SPORT_API_ID_CONTINUOUSGAIT = 1019
ROBOT_SPORT_API_ID_CONTENT = 1020
ROBOT_SPORT_API_ID_WALLOW = 1021
ROBOT_SPORT_API_ID_DANCE1 = 1022
ROBOT_SPORT_API_ID_DANCE2 = 1023
ROBOT_SPORT_API_ID_GETBODYHEIGHT = 1024
ROBOT_SPORT_API_ID_GETFOOTRAISEHEIGHT = 1025
ROBOT_SPORT_API_ID_GETSPEEDLEVEL = 1026
ROBOT_SPORT_API_ID_SWITCHJOYSTICK = 1027
ROBOT_SPORT_API_ID_POSE = 1028
ROBOT_SPORT_API_ID_SCRAPE = 1029
ROBOT_SPORT_API_ID_FRONTFLIP = 1030
ROBOT_SPORT_API_ID_FRONTJUMP = 1031
ROBOT_SPORT_API_ID_FRONTPOUNCE = 1032

# 机器人配置常量
class RobotConfig:
    """机器人配置常量"""
    num_dof = 12  # 自由度数量
    num_actions = 12  # 动作维度
    
    # 关节映射 The order of joints has been reindexed in simulation. So we do not need here.
    dof_map = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    # 关节名称
    dof_names = [
        "FR_hip_joint",    # 前右髋关节
        "FR_thigh_joint",  # 前右大腿关节
        "FR_calf_joint",   # 前右小腿关节
        "FL_hip_joint",    # 前左髋关节
        "FL_thigh_joint",  # 前左大腿关节
        "FL_calf_joint",   # 前左小腿关节
        "RR_hip_joint",    # 后右髋关节
        "RR_thigh_joint",  # 后右大腿关节
        "RR_calf_joint",   # 后右小腿关节
        "RL_hip_joint",    # 后左髋关节
        "RL_thigh_joint",  # 后左大腿关节
        "RL_calf_joint",   # 后左小腿关节
    ]
    
    # 关节方向符号
    dof_signs = [1.0] * 12
    
    # 关节限位 (高)
    joint_limits_high = [
        1.0472, 3.4907, -0.83776,  # FR
        1.0472, 3.4907, -0.83776,  # FL
        1.0472, 4.5379, -0.83776,  # RR
        1.0472, 4.5379, -0.83776,  # RL
    ]
    
    # 关节限位 (低)
    joint_limits_low = [
        -1.0472, -1.5708, -2.7227,  # FR
        -1.0472, -1.5708, -2.7227,  # FL
        -1.0472, -0.5236, -2.7227,  # RR
        -1.0472, -0.5236, -2.7227,  # RL
    ]
    
    # 扭矩限位
    torque_limits = [
        25, 40, 40,  # FR
        25, 40, 40,  # FL
        25, 40, 40,  # RR
        25, 40, 40,  # RL
    ]
    
    # 电机模式
    turn_on_motor_mode = [0x01] * 12

# 观察空间配置
class ObservationConfig:
    """观察空间配置"""
    n_proprio = 53        # 本体感受维度
    n_depth_latent = 32   # 深度特征维度
    n_hist_len = 10       # 历史长度

# 控制配置
class ControlConfig:
    """控制配置"""
    default_duration = 0.02  # 默认控制周期 (20ms)
    visual_update_interval = 5  # 视觉更新间隔

# 运行模式
class RunMode:
    """运行模式"""
    SPORT = "sport"           # 运动模式
    STAND = "stand"           # 站立模式
    LOCOMOTION = "locomotion" # 运动控制模式
    WALK = "walk"             # 行走模式 