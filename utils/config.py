import os.path as osp
import json
from collections import OrderedDict
import yaml
import re

def load_configuration(logdir):
    """
    Load configuration parameters from files and extract them for Go2ROS2Node initialization
    
    Args:
        logdir: Directory path containing the configuration file
        
    Returns:
        config_params: Dictionary with parameter names as keys for Go2ROS2Node constructor
        duration: Control cycle duration
    """
    
    if "EPO" in logdir:
        policy_source = "EPO"
    elif "legged-loco" in logdir:
        policy_source = "legged-loco"
    else:
        raise ValueError(f"Unknown policy source: {logdir}")
        
    # Extract the necessary parameters for Go2ROS2Node
    if policy_source == "EPO":
        config_path = osp.join(logdir, "config.json")
        with open(config_path, "r") as f:
            full_config = json.load(f, object_pairs_hook=OrderedDict)

        # EPO uses identity mapping (simulation order already matches hardware)
        joint_names = [
            "FR_hip_joint",     # 0
            "FR_thigh_joint",   # 1
            "FR_calf_joint",    # 2
            "FL_hip_joint",     # 3
            "FL_thigh_joint",   # 4
            "FL_calf_joint",    # 5
            "RR_hip_joint",     # 6
            "RR_thigh_joint",   # 7
            "RR_calf_joint",    # 8
            "RL_hip_joint",     # 9
            "RL_thigh_joint",   # 10
            "RL_calf_joint",    # 11
        ]
        obs_sources = ["proprio", "depth"]
        joint_map = get_joint_map_from_names(joint_names)
        default_joint_pos_dict = full_config.get("init_state", {}).get("default_joint_angles", {})
        default_joint_pos = parse_default_joint_pos_dict(default_joint_pos_dict, joint_names)
        kp = full_config.get("control", {}).get("stiffness", {}).get("joint", 40.)
        kd = full_config.get("control", {}).get("damping", {}).get("joint", 1.)
        obs_scales = full_config["normalization"]["obs_scales"]
        action_scale = full_config.get("control", {}).get("action_scale", 0.25)
        clip_obs = full_config["normalization"]["clip_obs"]
        clip_actions = full_config["normalization"]["clip_actions"]
        
    elif policy_source == "legged-loco":
        config_path = osp.join(logdir, "params/env.yaml")
        with open(config_path, "r") as f:
            full_config = yaml.safe_load(f)

        # legged-loco uses grouped joint order: [all hips, all thighs, all calves]
        # isaaclab sim: FL_hip(0), FR_hip(1), RL_hip(2), RR_hip(3), FL_thigh(4), FR_thigh(5), RL_thigh(6), RR_thigh(7), FL_calf(8), FR_calf(9), RL_calf(10), RR_calf(11)
        joint_names = [
            "FL_hip_joint",     # 0
            "FR_hip_joint",     # 1  
            "RL_hip_joint",     # 2
            "RR_hip_joint",     # 3
            "FL_thigh_joint",   # 4
            "FR_thigh_joint",   # 5
            "RL_thigh_joint",   # 6
            "RR_thigh_joint",   # 7
            "FL_calf_joint",    # 8
            "FR_calf_joint",    # 9
            "RL_calf_joint",    # 10
            "RR_calf_joint",    # 11
        ]
        # joint_map = [
        #     3, 0, 9, 6,   # Hip joints: FL->3, FR->0, RL->9, RR->6  
        #     4, 1, 10, 7,  # Thigh joints: FL->4, FR->1, RL->10, RR->7
        #     5, 2, 11, 8   # Calf joints: FL->5, FR->2, RL->11, RR->8
        # ]
        joint_map = get_joint_map_from_names(joint_names)
        default_joint_pos_dict = full_config.get("scene", {}).get("robot", {}).get("init_state", {}).get("joint_pos", {})
        default_joint_pos = parse_default_joint_pos_dict(default_joint_pos_dict, joint_names)
        kp = full_config.get("scene", {}).get("robot", {}).get("actuators", {}).get("base_legs", {}).get("stiffness", 40.0)
        kd = full_config.get("scene", {}).get("robot", {}).get("actuators", {}).get("base_legs", {}).get("damping", 1.0)
        obs_scales = {}
        action_scale = full_config.get("actions", {}).get("joint_pos", {}).get("scale", 0.25)
        clip_obs = 100.0
        clip_actions = None
    else:
        raise ValueError(f"Unknown policy source: {policy_source}")

    # Set control cycle = dt * decimation = 0.005 * 4 = 0.02 (fixed at 20ms)
    duration = 0.02

    return joint_map, default_joint_pos, kp, kd, obs_scales, action_scale, clip_obs, clip_actions, duration


def get_joint_map_from_names(joint_names):
    """
    Generate joint map from simulation order to hardware order based on joint names.
    
    Hardware order: FR_hip(0), FR_thigh(1), FR_calf(2), FL_hip(3), FL_thigh(4), FL_calf(5), 
                   RR_hip(6), RR_thigh(7), RR_calf(8), RL_hip(9), RL_thigh(10), RL_calf(11)
    
    Args:
        joint_names: List of joint names in simulation order
        
    Returns:
        joint_map: List where each element is the hardware index for the corresponding simulation index
    """
    # Define hardware order mapping: joint_name -> hardware_index
    hardware_order = {
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
    
    # Generate joint map: simulation_index -> hardware_index
    joint_map = []
    for joint_name in joint_names:
        if joint_name not in hardware_order:
            raise ValueError(f"Unknown joint name: {joint_name}. Valid joint names are: {list(hardware_order.keys())}")
        joint_map.append(hardware_order[joint_name])
    
    return joint_map

def parse_default_joint_pos_dict(default_joint_pos_dict, joint_names):
    """
    Parse default joint positions from the dictionary (name: value) to list of values in simulation order

    Args:
        default_joint_pos_dict: Dictionary of default joint positions (name or regex pattern: value)
        joint_names: List of joint names in simulation order
        
    Returns:
        default_joint_pos: List of default joint positions in simulation order
    """
    # Check if this is a regex pattern dictionary (legged-loco) or direct names (EPO)
    # If any key contains regex special characters, treat as regex patterns
    has_regex_patterns = any(
        any(char in key for char in ['.*', '[', ']', '(', ')', '+', '*', '?', '^', '$'])
        for key in default_joint_pos_dict.keys()
    )
    
    if has_regex_patterns:
        # Expand regex patterns to actual joint names
        expanded_dict = expand_regex_joint_positions(default_joint_pos_dict, joint_names)
        default_joint_pos_dict = expanded_dict
    
    # Convert to list in simulation order
    default_joint_pos = []
    for i in range(len(joint_names)):
        if joint_names[i] in default_joint_pos_dict:
            default_joint_pos.append(default_joint_pos_dict[joint_names[i]])
        else:
            raise KeyError(f"Joint '{joint_names[i]}' not found in default joint positions")
    
    return default_joint_pos


def expand_regex_joint_positions(regex_joint_pos_dict, joint_names):
    """
    Expand regex patterns in legged-loco joint positions to actual joint names
    
    Args:
        regex_joint_pos_dict: Dictionary with regex patterns as keys and joint positions as values
        joint_names: List of actual joint names to match against
        
    Returns:
        expanded_dict: Dictionary with actual joint names as keys and positions as values
    """
    expanded_dict = {}
    
    for pattern, value in regex_joint_pos_dict.items():
        # Find all joint names that match this pattern
        for joint_name in joint_names:
            if re.fullmatch(pattern, joint_name):
                expanded_dict[joint_name] = value
    
    return expanded_dict