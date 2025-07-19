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