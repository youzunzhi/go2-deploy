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

# 运动模式切换API ID (Go2 1.1.7+)
MOTION_SWITCHER_API_ID_RELEASE = 1003  # 释放模式，切换到普通模式
MOTION_SWITCHER_API_ID_SELECT_MCF = 1002  # 选择MCF模式

# 手柄按钮定义
class WirelessButtons:
    """无线手柄按钮定义"""
    L1 = 0x0001
    L2 = 0x0002
    R1 = 0x0004
    R2 = 0x0008
    X = 0x0010
    Y = 0x0020
    A = 0x0040
    B = 0x0080
    UP = 0x0100
    DOWN = 0x0200
    LEFT = 0x0400
    RIGHT = 0x0800
    SELECT = 0x1000
    START = 0x2000
    HOME = 0x4000
    L3 = 0x8000
    R3 = 0x10000

# 机器人配置
class RobotConfig:
    """机器人配置"""
    num_dof = 12
    num_actions = 12
    dof_map = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    dof_names = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", 
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"
    ]
    dof_signs = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    turn_on_motor_mode = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    
    # 关节限制
    joint_limits_low = [-0.802851, -1.0472, -2.69653, -0.802851, -1.0472, -2.69653,
                       -0.802851, -1.0472, -2.69653, -0.802851, -1.0472, -2.69653]
    joint_limits_high = [0.802851, 4.18879, -0.916298, 0.802851, 4.18879, -0.916298,
                        0.802851, 4.18879, -0.916298, 0.802851, 4.18879, -0.916298]

# 观察空间配置
class ObservationConfig:
    """观察空间配置"""
    n_proprio = 53
    n_depth_latent = 32
    n_hist_len = 10

# 控制配置
class ControlConfig:
    """控制配置"""
    visual_update_interval = 5
    lin_vel_deadband = 0.1
    ang_vel_deadband = 0.1
    cmd_px_range = [0.4, 1.0]
    cmd_nx_range = [0.4, 0.8]
    cmd_py_range = [0.4, 0.8]
    cmd_ny_range = [0.4, 0.8]
    cmd_pyaw_range = [0.4, 1.6]
    cmd_nyaw_range = [0.4, 1.6]

# 运行模式
class RunMode:
    """运行模式"""
    SPORT = "sport"           # 运动模式
    STAND = "stand"           # 站立模式
    LOCOMOTION = "locomotion" # 运动控制模式
    WALK = "walk"             # 行走模式 