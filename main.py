import rclpy
from rclpy.node import Node
from go2_ros2_node import Go2ROS2Node, get_euler_xyz

import os
import ast
import os.path as osp
import json
import time
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import torch.jit

from rsl_rl import modules
from rsl_rl.modules import StateHistoryEncoder, RecurrentDepthBackbone, DepthOnlyFCBackbone58x87
import cv2

import sys
import time
import sys
import threading


ROBOT_SPORT_API_ID_BALANCESTAND = 1002
ROBOT_SPORT_API_ID_STANDUP = 1004
ROBOT_SPORT_API_ID_STANDDOWN = 1005


def start_main_loop_timer(node, duration):
    """Start the main loop timer for ROS-based timing control"""
    node.main_loop_timer = node.create_timer(
        duration, # in sec
        lambda: main_loop(node),
    )


def main_loop(node):
    """Main control loop for the Go2 robot - handles different operational modes based on joystick input"""
    if node.use_sport_mode:
        if (node.joy_stick_buffer.keys & node.WirelessButtons.R1):
            node.get_logger().info("In the sport mode, R1 pressed, robot will stand up.")
            node._sport_mode_change(ROBOT_SPORT_API_ID_STANDUP)
        if (node.joy_stick_buffer.keys & node.WirelessButtons.R2):
            node.get_logger().info("In the sport mode, R2 pressed, robot will sit down.")
            node._sport_mode_change(ROBOT_SPORT_API_ID_STANDDOWN)

        if (node.joy_stick_buffer.keys & node.WirelessButtons.X):
            node.get_logger().info("In the sport mode, X pressed, robot will balance stand.")
            node._sport_mode_change(ROBOT_SPORT_API_ID_BALANCESTAND)

        if (node.joy_stick_buffer.keys & node.WirelessButtons.L1):
            node.get_logger().info("Exist the sport mode. Switch to stand policy.")
            node.use_sport_mode = False
            node._sport_state_change(0)
            node.use_stand_policy = True
            node.use_locomotion_policy = False
    
    if node.use_stand_policy:
        stand_action = node.get_stand_action()
        node.send_stand_action(stand_action)
    
    if (node.joy_stick_buffer.keys & node.WirelessButtons.Y):
        node.get_logger().info("Y pressed, use the locomotion policy")
        node.use_stand_policy = False
        node.use_locomotion_policy = True
        node.use_sport_mode = False
        node.global_counter = 0

    if node.use_locomotion_policy:
        node.use_stand_policy = False
        node.use_sport_mode = False
        
        # Handle X button for legged-loco policy - set forward command
        if (node.joy_stick_buffer.keys & node.WirelessButtons.X):
            if node.policy_source == "legged-loco":
                node.get_logger().info("X pressed, setting legged-loco command to [0.4, 0, 0]")
                node.xyyaw_command = torch.tensor([[0.4, 0.0, 0.0]], device=node.model_device, dtype=torch.float32)
        
        start_time = time.monotonic()

        proprio = node.get_proprio()
        get_pro_time = time.monotonic()

        proprio_history = node._get_history_proprio()
        get_hist_pro_time = time.monotonic()

        # print('proprioception: ', proprio)
        # print('history proprioception: ', proprio_history)

        if node.global_counter % node.visual_update_interval == 0:
            depth_image = node._get_depth_image()
            if node.global_counter == 0:
                node.last_depth_image = depth_image
            node.depth_latent_yaw = node.depth_encode(node.last_depth_image, proprio)
            node.last_depth_image = depth_image
            # print('depth latent: ', node.depth_latent_yaw)
        get_obs_time = time.monotonic()

        obs = node.turn_obs(proprio, node.depth_latent_yaw, proprio_history, node.n_proprio, node.n_depth_latent, node.n_hist_len)
        turn_obs_time = time.monotonic()

        action = node.policy(obs)
        policy_time = time.monotonic()
        # print('action before clip and normalize: ', action)

        node.send_action(action)
        print('action: ', action)

        publish_time = time.monotonic()
        print(
            "get proprio time: {:.5f}".format(get_pro_time - start_time),
            "get hist pro time: {:.5f}".format(get_hist_pro_time - get_pro_time),
            "get_depth time: {:.5f}".format(get_obs_time - get_hist_pro_time),
            "get obs time: {:.5f}".format(get_obs_time - start_time),
            "turn_obs_time: {:.5f}".format(turn_obs_time - get_obs_time),
            "policy_time: {:.5f}".format(policy_time - turn_obs_time),
            "publish_time: {:.5f}".format(publish_time - policy_time),
            "total time: {:.5f}".format(publish_time - start_time)
        )

        node.global_counter += 1

    if (node.joy_stick_buffer.keys & node.WirelessButtons.L2):
        node.get_logger().info("L2 pressed, stop using locomotion policy, switch to sport mode.")
        node.use_stand_policy = False
        node.use_locomotion_policy = False
        node.use_sport_mode = True
        node.reset_obs()
        node._sport_state_change(1)
        node._sport_mode_change(ROBOT_SPORT_API_ID_BALANCESTAND)


def handle_timing_mode(env_node, timing_mode, duration):
    if timing_mode == "ros_timer":
        # Use ROS timer for timing control
        env_node.get_logger().info('Model and Policy are ready')
        start_main_loop_timer(env_node, duration)
        rclpy.spin(env_node)
    
    elif timing_mode == "manual_control":
        # Manually control timing for more precise control
        rclpy.spin_once(env_node, timeout_sec=0.)
        env_node.get_logger().info("Model and Policy are ready")
        
        while rclpy.ok():
            # Track iteration time to maintain desired frequency
            main_loop_time = time.monotonic()
            
            # Run one iteration
            main_loop(env_node)
            rclpy.spin_once(env_node, timeout_sec=0.)
            
            # Sleep remaining time to maintain frequency
            sleep_time = max(0, duration - (time.monotonic() - main_loop_time))
            time.sleep(sleep_time)
    
    else:
        raise ValueError(f"Invalid timing mode: {timing_mode}")


def load_configuration(logdir):
    """
    Load only necessary configuration parameters for deployment
    
    Args:
        logdir: Directory path containing the configuration file
        
    Returns:
        config_dict: Filtered configuration dictionary with only used parameters
        duration: Control cycle duration
    """
    assert logdir is not None, "Please provide a logdir"
    
    # Load full training configuration file
    config_path = osp.join(logdir, "config.json")
    with open(config_path, "r") as f:
        full_config = json.load(f, object_pairs_hook=OrderedDict)
    
    # Determine policy source based on config structure
    if "control" in full_config and "control_type" in full_config["control"]:
        policy_source = "EPO"
    elif "scene" in full_config and "robot" in full_config["scene"]:
        policy_source = "legged-loco"
    else:
        policy_source = "unknown"
    
    # Extract only the necessary parameters that are actually used
    if policy_source == "EPO":
        # EPO uses observation scaling from config.json
        config_dict = {
            "normalization": {
                "clip_observations": full_config["normalization"]["clip_observations"],
                "clip_actions": full_config["normalization"]["clip_actions"],
                "obs_scales": {
                    "ang_vel": full_config["normalization"]["obs_scales"]["ang_vel"],
                    "dof_pos": full_config["normalization"]["obs_scales"]["dof_pos"],
                    "dof_vel": full_config["normalization"]["obs_scales"]["dof_vel"]
                }
            }
        }
        
        # Handle optional clip_actions_method parameter
        if "clip_actions_method" in full_config["normalization"]:
            config_dict["normalization"]["clip_actions_method"] = full_config["normalization"]["clip_actions_method"]
            
            # Add clip_actions_high and clip_actions_low if hard clipping is used
            if full_config["normalization"].get("clip_actions_method") == "hard":
                config_dict["normalization"]["clip_actions_high"] = full_config["normalization"]["clip_actions_high"]
                config_dict["normalization"]["clip_actions_low"] = full_config["normalization"]["clip_actions_low"]
        
        # EPO stores control parameters under "control" key
        control_config = full_config.get("control", {})
        config_dict.update({
            "control": {
                "control_type": control_config.get("control_type", "P"),
                "stiffness": control_config.get("stiffness", {}),
                "damping": control_config.get("damping", {}),
                "action_scale": control_config.get("action_scale", 0.25),
            },
            "init_state": {
                "default_joint_angles": full_config.get("init_state", {}).get("default_joint_angles", {})
            }
        })
        
    elif policy_source == "legged-loco":
        # legged-loco does NOT use observation scaling (confirmed from training code)
        # All scale values are null in env.yaml and empirical_normalization: false
        config_dict = {
            "normalization": {
                "clip_observations": 100.0,  # Default fallback
                "clip_actions": 100.0,  # Default fallback, legged-loco doesn't have this
                "obs_scales": {
                    "ang_vel": 1.0,  # No scaling in legged-loco training
                    "dof_pos": 1.0,  # No scaling in legged-loco training
                    "dof_vel": 1.0   # No scaling in legged-loco training
                }
            }
        }
        
        # legged-loco stores control parameters in scene.robot.actuators.base_legs
        actuator_config = full_config.get("scene", {}).get("robot", {}).get("actuators", {}).get("base_legs", {})
        action_config = full_config.get("actions", {}).get("joint_pos", {})
        init_state_config = full_config.get("scene", {}).get("robot", {}).get("init_state", {})
        
        # Extract joint positions from legged-loco config
        joint_pos = init_state_config.get("joint_pos", {})
        
        # If joint_pos is empty, use EPO's default joint angles (identical values)
        if not joint_pos:
            joint_pos = {
                "FL_hip_joint": 0.1,
                "RL_hip_joint": 0.1,
                "FR_hip_joint": -0.1,
                "RR_hip_joint": -0.1,
                "FL_thigh_joint": 0.8,
                "RL_thigh_joint": 1.0,
                "FR_thigh_joint": 0.8,
                "RR_thigh_joint": 1.0,
                "FL_calf_joint": -1.5,
                "RL_calf_joint": -1.5,
                "FR_calf_joint": -1.5,
                "RR_calf_joint": -1.5
            }
        
        config_dict.update({
            "control": {
                "control_type": "P",  # Default for legged-loco
                "stiffness": actuator_config.get("stiffness", 40.0),  # Default from legged-loco
                "damping": actuator_config.get("damping", 1.0),      # Default from legged-loco
                "action_scale": action_config.get("scale", 0.25),    # Default from legged-loco
            },
            "init_state": {
                "default_joint_angles": joint_pos
            }
        })
    else:
        raise ValueError(f"Unknown policy source: {policy_source}")
    
    # Set control cycle (fixed at 20ms, different from training)
    duration = 0.02
    
    return config_dict, duration


def load_base_model(logdir, device):
    """
    Load base model (JIT format)
    
    Args:
        logdir: Model file directory
        device: Computing device (cuda/cpu)
        
    Returns:
        base_model: Loaded base model
        estimator: Speed estimator
        hist_encoder: History encoder
        actor: Action generator
    """
    base_model_name = 'base_jit.pt'
    base_model_path = os.path.join(logdir, base_model_name)
    
    # Load base model in JIT format
    base_model = torch.jit.load(base_model_path, map_location=device)
    base_model.eval()
    
    # Extract model components
    estimator = base_model.estimator.estimator
    hist_encoder = base_model.actor.history_encoder
    actor = base_model.actor.actor_backbone
    
    return base_model, estimator, hist_encoder, actor


def load_vision_model(logdir, device):
    """
    Load vision model (depth encoder)
    
    Args:
        logdir: Model file directory
        device: Computing device
        
    Returns:
        depth_encoder: Depth encoder model
    """
    vision_model_name = 'vision_weight.pt'
    vision_model_path = os.path.join(logdir, vision_model_name)
    
    # Load vision model weights
    vision_model = torch.load(vision_model_path, map_location=device)
    
    # Create depth encoder
    depth_backbone = DepthOnlyFCBackbone58x87(None, 32, 512)
    depth_encoder = RecurrentDepthBackbone(depth_backbone, None).to(device)
    
    # Load pre-trained weights
    depth_encoder.load_state_dict(vision_model['depth_encoder_state_dict'])
    depth_encoder.to(device)
    depth_encoder.eval()
    
    return depth_encoder


def create_observation_processor(estimator, hist_encoder):
    """
    Create observation processing function
    
    Args:
        estimator: Speed estimator
        hist_encoder: History encoder
        
    Returns:
        turn_obs: Observation processing function
    """
    def turn_obs(proprio, depth_latent_yaw, proprio_history, n_proprio, n_depth_latent, n_hist_len):
        """
        Convert raw sensor data to neural network input format
        
        Args:
            proprio: Proprioceptive data
            depth_latent_yaw: Depth features and yaw angle
            proprio_history: Historical proprioceptive data
            n_proprio: Proprioceptive data dimension
            n_depth_latent: Depth feature dimension
            n_hist_len: History length
            
        Returns:
            obs: Processed observation vector
        """
        # Separate depth features and yaw angle
        depth_latent = depth_latent_yaw[:, :-2]
        yaw = depth_latent_yaw[:, -2:] * 1.5
        print('yaw: ', yaw)
        
        # Update yaw angle in proprioceptive data
        proprio[:, 6:8] = yaw
        
        # Estimate linear velocity features
        lin_vel_latent = estimator(proprio)
        
        # Process historical proprioceptive data
        activation = nn.ELU()
        priv_latent = hist_encoder(activation, proprio_history.view(-1, n_hist_len, n_proprio))
        
        # Concatenate all features
        obs = torch.cat([proprio, depth_latent, lin_vel_latent, priv_latent], dim=-1)
        
        return obs
    
    return turn_obs


def create_depth_encoder(depth_encoder):
    """
    Create depth encoding function
    
    Args:
        depth_encoder: Depth encoder model
        
    Returns:
        encode_depth: Depth encoding function
    """
    def encode_depth(depth_image, proprio):
        """
        Encode depth image into feature vector
        
        Args:
            depth_image: Depth image
            proprio: Proprioceptive data
            
        Returns:
            depth_latent_yaw: Depth features and yaw angle
        """
        depth_latent_yaw = depth_encoder(depth_image, proprio)
        
        # Check for NaN values
        if torch.isnan(depth_latent_yaw).any():
            print('depth_latent_yaw contains nan and the depth image is: ', depth_image)
        
        return depth_latent_yaw
    
    return encode_depth


def create_policy(actor):
    """
    Create policy function
    
    Args:
        actor: Action generator
        
    Returns:
        actor_model: Policy function
    """
    def actor_model(obs):
        """
        Generate action based on observation
        
        Args:
            obs: Observation vector
            
        Returns:
            action: Action vector
        """
        action = actor(obs)
        return action
    
    return actor_model


def log_system_info(env_node, logdir, duration):
    """
    Print system configuration information
    
    Args:
        env_node: ROS node
        logdir: Model directory
        duration: Control cycle
    """
    env_node.get_logger().info("Model loaded from: {}".format(osp.join(logdir)))
    env_node.get_logger().info("Control Duration: {} sec".format(duration))
    env_node.get_logger().info("Motor Stiffness (kp): {}".format(env_node.p_gains))
    env_node.get_logger().info("Motor Damping (kd): {}".format(env_node.d_gains))


def setup_models(logdir, device):
    """
    Unified setup of models and inference functions
    
    Args:
        logdir: Model file directory
        device: Computing device
        
    Returns:
        turn_obs: Observation processing function
        encode_depth: Depth encoding function
        actor_model: Policy function
    """
    # Load base model
    base_model, estimator, hist_encoder, actor = load_base_model(logdir, device)
    
    # Load vision model
    depth_encoder = load_vision_model(logdir, device)
    
    # Create inference functions
    turn_obs = create_observation_processor(estimator, hist_encoder)
    encode_depth = create_depth_encoder(depth_encoder)
    actor_model = create_policy(actor)
    
    return turn_obs, encode_depth, actor_model


@torch.inference_mode()
def main(args):
    rclpy.init()

    # 1. Load configuration
    config_dict, duration = load_configuration(args.logdir)
    device = "cuda"

    # 2. Create ROS node
    env_node = Go2ROS2Node(
        "go2",
        cfg=config_dict,
        model_device=device,
        dryrun=not args.nodryrun,
        mode=args.mode,
    )

    # 3. Print configuration information
    log_system_info(env_node, args.logdir, duration)

    # 4. Load models and create inference functions
    turn_obs, encode_depth, actor_model = setup_models(args.logdir, device)

    # 5. Register models to node
    env_node.register_models(turn_obs=turn_obs, depth_encode=encode_depth, policy=actor_model)
    env_node.start_ros_handlers()
    env_node.warm_up()

    # 6. Start control loop
    handle_timing_mode(env_node, args.timing_mode, duration)

    rclpy.shutdown()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--logdir", type= str, default= None, help= "The directory which contains the config.json and model_*.pt files")
    parser.add_argument("--nodryrun", action= "store_true", default= False, help= "Disable dryrun mode")
    parser.add_argument("--timing_mode", type=str, default="ros_timer",
        choices=["manual_control", "ros_timer"],
        help="Select timing mode: manual_control (precise timing control) or ros_timer (ROS managed timer)",
    )
    parser.add_argument("--mode", type= str, default= "locomotion", choices=["locomotion", "walk"])
    args = parser.parse_args()
    
    main(args)
