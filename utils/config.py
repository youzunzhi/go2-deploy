import os.path as osp
import json
from collections import OrderedDict
import yaml
import re
from .hardware import HARDWARE_ORDER

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

    
    # Generate joint map: simulation_index -> hardware_index
    joint_map = []
    for joint_name in joint_names:
        if joint_name not in HARDWARE_ORDER:
            raise ValueError(f"Unknown joint name: {joint_name}. Valid joint names are: {list(HARDWARE_ORDER.keys())}")
        joint_map.append(HARDWARE_ORDER[joint_name])
    
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