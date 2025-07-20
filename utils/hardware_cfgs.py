class WirelessButtons:
    R1 =            0b00000001 # 1
    L1 =            0b00000010 # 2
    start =         0b00000100 # 4
    select =        0b00001000 # 8
    R2 =            0b00010000 # 16
    L2 =            0b00100000 # 32
    F1 =            0b01000000 # 64
    F2 =            0b10000000 # 128
    A =             0b100000000 # 256
    B =             0b1000000000 # 512
    X =             0b10000000000 # 1024
    Y =             0b100000000000 # 2048
    up =            0b1000000000000 # 4096
    right =         0b10000000000000 # 8192
    down =          0b100000000000000 # 16384
    left =          0b1000000000000000 # 32768

# ROS2 Topic Names
ROS_TOPICS = {
    "LOW_STATE": "/lowstate",
    "LOW_CMD": "/lowcmd",
    "WIRELESS_CONTROLLER": "/wirelesscontroller",
    "DEPTH_IMAGE": "/forward_depth_image",
    "SPORT_MODE": "/api/sport_mode/request",
    "MOTION_SWITCHER": "/api/motion_switcher/request",
}

HARDWARE_ORDER = {
    "FR_hip_joint": 0,
    "FR_thigh_joint": 1,
    "FR_calf_joint": 2,
    "FL_hip_joint": 3,
    "FL_thigh_joint": 4,
    "FL_calf_joint": 5,
    "RR_hip_joint": 6,
    "RR_thigh_joint": 7,
    "RR_calf_joint": 8,
    "RL_hip_joint": 9,
    "RL_thigh_joint": 10,
    "RL_calf_joint": 11,
}

# --- Joint Limits ---
JOINT_POS_LIMIT_HIGH = {
    "FR_hip_joint": 1.0472,
    "FR_thigh_joint": 3.4907,
    "FR_calf_joint": -0.83776,
    "FL_hip_joint": 1.0472,
    "FL_thigh_joint": 3.4907,
    "FL_calf_joint": -0.83776,
    "RR_hip_joint": 1.0472,
    "RR_thigh_joint": 4.5379,
    "RR_calf_joint": -0.83776,
    "RL_hip_joint": 1.0472,
    "RL_thigh_joint": 4.5379,
    "RL_calf_joint": -0.83776,
}

JOINT_POS_LIMIT_LOW = {
    "FR_hip_joint": -1.0472,
    "FR_thigh_joint": -1.5708,
    "FR_calf_joint": -2.7227,
    "FL_hip_joint": -1.0472,
    "FL_thigh_joint": -1.5708,
    "FL_calf_joint": -2.7227,
    "RR_hip_joint": -1.0472,
    "RR_thigh_joint": -0.5236,
    "RR_calf_joint": -2.7227,
    "RL_hip_joint": -1.0472,
    "RL_thigh_joint": -0.5236,
    "RL_calf_joint": -2.7227,
}

TORQUE_LIMIT = {
    "FR_hip_joint": 25,
    "FR_thigh_joint": 40,
    "FR_calf_joint": 40,
    "FL_hip_joint": 25,
    "FL_thigh_joint": 40,
    "FL_calf_joint": 40,
    "RR_hip_joint": 25,
    "RR_thigh_joint": 40,
    "RR_calf_joint": 40,
    "RL_hip_joint": 25,
    "RL_thigh_joint": 40,
    "RL_calf_joint": 40,
}


def get_joint_limits_in_sim_order(joint_map, device):
    """Setup joint position and torque limits in simulation order."""
    import torch
    from .joint_order_util import map_list_in_real_order_to_sim_order
    
    # Use policy-specific joint and torque limits
    joint_pos_limit_high_real = list(JOINT_POS_LIMIT_HIGH.values())
    joint_pos_limit_low_real = list(JOINT_POS_LIMIT_LOW.values())
    torque_limit_real = list(TORQUE_LIMIT.values())
    joint_pos_limit_high_sim = map_list_in_real_order_to_sim_order(joint_pos_limit_high_real, joint_map)
    joint_pos_limit_low_sim = map_list_in_real_order_to_sim_order(joint_pos_limit_low_real, joint_map)
    torque_limit_sim = map_list_in_real_order_to_sim_order(torque_limit_real, joint_map)
    joint_pos_limit_high_sim = torch.tensor(joint_pos_limit_high_sim, device=device, dtype=torch.float32)
    joint_pos_limit_low_sim = torch.tensor(joint_pos_limit_low_sim, device=device, dtype=torch.float32)
    torque_limit_sim = torch.tensor(torque_limit_sim, device=device, dtype=torch.float32)
    
    return joint_pos_limit_high_sim, joint_pos_limit_low_sim, torque_limit_sim

# --------------------
# --- For Stand Up ---
STAND_TARGET_POS_STAGE1 = {
    "FR_hip_joint": 0.0,
    "FR_thigh_joint": 1.36,
    "FR_calf_joint": -2.65,
    "FL_hip_joint": 0.0,
    "FL_thigh_joint": 1.36,
    "FL_calf_joint": -2.65,
    "RR_hip_joint": 0.0,
    "RR_thigh_joint": 1.36,
    "RR_calf_joint": -2.65,
    "RL_hip_joint": 0.0,
    "RL_thigh_joint": 1.36,
    "RL_calf_joint": -2.65,
}

STAND_TARGET_POS_STAGE2 = {
    "FR_hip_joint": 0.0,
    "FR_thigh_joint": 0.67,
    "FR_calf_joint": -1.3,
    "FL_hip_joint": 0.0,
    "FL_thigh_joint": 0.67,
    "FL_calf_joint": -1.3,
    "RR_hip_joint": 0.0,
    "RR_thigh_joint": 0.67,
    "RR_calf_joint": -1.3,
    "RL_hip_joint": 0.0,
    "RL_thigh_joint": 0.67,
    "RL_calf_joint": -1.3,
}

STAND_STAGE1_DURATION = 10
STAND_STAGE2_DURATION = 100
