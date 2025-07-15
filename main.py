import rclpy
from rclpy.node import Node
from go2_ros2_node import Go2Handler, get_euler_xyz

import os
import ast
import os.path as osp
import time
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import torch.jit
import yaml
import re

from rsl_rl import modules
from rsl_rl.modules import StateHistoryEncoder, RecurrentDepthBackbone, DepthOnlyFCBackbone58x87
import cv2

import sys
import time
import sys
import threading

from utils import load_configuration

ROBOT_SPORT_API_ID_BALANCESTAND = 1002
ROBOT_SPORT_API_ID_STANDUP = 1004
ROBOT_SPORT_API_ID_STANDDOWN = 1005


def start_main_loop_timer(handler, duration):
    """Start the main loop timer for ROS-based timing control"""
    handler.main_loop_timer = handler.node.create_timer(
        duration, # in sec
        lambda: main_loop(handler),
    )


def main_loop(handler):
    """Main control loop for the Go2 robot - handles different operational modes based on joystick input"""
    if handler.use_sport_mode:
        if (handler.joy_stick_buffer.keys & handler.WirelessButtons.R1):
            handler.node.get_logger().info("In the sport mode, R1 pressed, robot will stand up.")
            handler._sport_mode_change(ROBOT_SPORT_API_ID_STANDUP)
        if (handler.joy_stick_buffer.keys & handler.WirelessButtons.R2):
            handler.node.get_logger().info("In the sport mode, R2 pressed, robot will sit down.")
            handler._sport_mode_change(ROBOT_SPORT_API_ID_STANDDOWN)

        if (handler.joy_stick_buffer.keys & handler.WirelessButtons.X):
            handler.node.get_logger().info("In the sport mode, X pressed, robot will balance stand.")
            handler._sport_mode_change(ROBOT_SPORT_API_ID_BALANCESTAND)

        if (handler.joy_stick_buffer.keys & handler.WirelessButtons.L1):
            handler.node.get_logger().info("Exist the sport mode. Switch to stand policy.")
            handler.use_sport_mode = False
            handler._sport_state_change(0)
            handler.use_stand_policy = True
            handler.use_locomotion_policy = False
    
    if handler.use_stand_policy:
        stand_action = handler.get_stand_action()
        handler.send_stand_action(stand_action)
    
    if (handler.joy_stick_buffer.keys & handler.WirelessButtons.Y):
        handler.node.get_logger().info("Y pressed, use the locomotion policy")
        handler.use_stand_policy = False
        handler.use_locomotion_policy = True
        handler.use_sport_mode = False
        handler.global_counter = 0

    if handler.use_locomotion_policy:
        handler.use_stand_policy = False
        handler.use_sport_mode = False
        
        # Handle X button for legged-loco policy - set forward command
        if (handler.joy_stick_buffer.keys & handler.WirelessButtons.X):
            if handler.policy_source == "legged-loco":
                handler.node.get_logger().info("X pressed, setting legged-loco command to [0.4, 0, 0]")
                handler.xyyaw_command = torch.tensor([[0.4, 0.0, 0.0]], device=handler.device, dtype=torch.float32)
        
        proprio = handler.get_proprio()
        proprio_history = handler._get_history_proprio()
        if handler.global_counter % handler.visual_update_interval == 0:
            depth_image = handler._get_depth_image()
            if handler.global_counter == 0:
                handler.last_depth_image = depth_image
            handler.depth_latent_yaw = handler.depth_encode(handler.last_depth_image, proprio)
            handler.last_depth_image = depth_image

        obs = handler.turn_obs(proprio, handler.depth_latent_yaw, proprio_history, handler.n_proprio, handler.n_depth_latent, handler.n_hist_len)

        action = handler.policy(obs)

        handler.send_action(action)

        handler.global_counter += 1

    if (handler.joy_stick_buffer.keys & handler.WirelessButtons.L2):
        handler.node.get_logger().info("L2 pressed, stop using locomotion policy, switch to sport mode.")
        handler.use_stand_policy = False
        node.use_locomotion_policy = False
        node.use_sport_mode = True
        handler.reset_obs()
        handler._sport_state_change(1)
        handler._sport_mode_change(ROBOT_SPORT_API_ID_BALANCESTAND)


def handle_timing_mode(handler, timing_mode, duration):
    if timing_mode == "ros_timer":
        # Use ROS timer for timing control
        handler.node.get_logger().info('Model and Policy are ready')
        start_main_loop_timer(handler, duration)
        rclpy.spin(handler.node)
    
    elif timing_mode == "manual_control":
        # Manually control timing for more precise control
        rclpy.spin_once(handler.node, timeout_sec=0.)
        handler.node.get_logger().info("Model and Policy are ready")
        
        while rclpy.ok():
            # Track iteration time to maintain desired frequency
            main_loop_time = time.monotonic()
            
            # Run one iteration
            main_loop(handler)
            rclpy.spin_once(handler.node, timeout_sec=0.)
            
            # Sleep remaining time to maintain frequency
            sleep_time = max(0, duration - (time.monotonic() - main_loop_time))
            time.sleep(sleep_time)
    
    else:
        raise ValueError(f"Invalid timing mode: {timing_mode}")


def create_observation_processor(estimator, hist_encoder):
    """
    Create observation processing function
    
    Args:
        estimator: Speed estimator
        hist_encoder: History encoder
        
    Returns:
        turn_obs: Observation processing function
    """

    
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


def log_system_info(handler, logdir, duration):
    """
    Print system configuration information
    
    Args:
        handler: Go2 Handler
        logdir: Model directory
        duration: Control cycle
    """
    handler.node.get_logger().info("Model loaded from: {}".format(osp.join(logdir)))
    handler.node.get_logger().info("Control Duration: {} sec".format(duration))
    handler.node.get_logger().info("Motor Stiffness (kp): {}".format(handler.kp))
    handler.node.get_logger().info("Motor Damping (kd): {}".format(handler.kd))


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
    # Determine policy source from logdir structure
    policy_source = determine_policy_source(logdir)
    
    if policy_source == "EPO":
        # EPO: Load separate base model and vision model
        base_model, estimator, hist_encoder, actor = load_base_model(logdir, device)
        depth_encoder = load_vision_model(logdir, device)
        
        # Create inference functions for EPO
        turn_obs = create_observation_processor(estimator, hist_encoder)
        encode_depth = create_depth_encoder(depth_encoder)
        actor_model = create_policy(actor)
        
    elif policy_source == "legged-loco":
        # legged-loco: Load single JIT policy
        policy = load_legged_loco_policy(logdir, device)
        
        # Create inference functions for legged-loco
        turn_obs = create_legged_loco_observation_processor()
        encode_depth = create_dummy_depth_encoder()  # No vision in base policy
        actor_model = create_legged_loco_policy_function(policy)
        
    else:
        raise ValueError(f"Unsupported policy source: {policy_source}")
    
    return turn_obs, encode_depth, actor_model


def load_legged_loco_policy(logdir, device):
    """
    Load legged-loco policy (single JIT model)
    
    Args:
        logdir: Model file directory
        device: Computing device
        
    Returns:
        policy: Loaded policy model
    """
    policy_path = os.path.join(logdir, 'policy.jit')
    
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    
    # Load JIT policy
    policy = torch.jit.load(policy_path, map_location=device)
    policy.eval()
    
    return policy


def create_legged_loco_observation_processor():
    """
    Create observation processing function for legged-loco policy
    
    Returns:
        turn_obs: Observation processing function
    """
    def turn_obs(proprio, depth_latent_yaw, proprio_history, n_proprio, n_depth_latent, n_hist_len):
        """
        Process observations for legged-loco policy
        
        Args:
            proprio: Current proprioceptive data [batch_size, 45]
            depth_latent_yaw: Depth features (unused for base policy)
            proprio_history: Historical proprioceptive data [batch_size, hist_len, 45]
            n_proprio: Proprioceptive data dimension (45)
            n_depth_latent: Depth feature dimension (unused)
            n_hist_len: History length (9)
            
        Returns:
            obs: Processed observation vector [batch_size, 450]
        """
        # legged-loco expects: [current_obs, hist_obs_t-1, hist_obs_t-2, ..., hist_obs_t-9]
        # Shape: [batch_size, 45 * 10] = [batch_size, 450]
        
        # Reshape history: [batch_size, hist_len, n_proprio] -> [batch_size, hist_len * n_proprio]
        flattened_history = proprio_history.view(proprio_history.shape[0], -1)
        
        # Concatenate current observation with flattened history
        obs = torch.cat([proprio, flattened_history], dim=-1)
        
        return obs
    
    return turn_obs


def create_dummy_depth_encoder():
    """
    Create dummy depth encoder for legged-loco base policy (no vision)
    
    Returns:
        encode_depth: Dummy depth encoding function
    """
    def encode_depth(depth_image, proprio):
        """
        Dummy depth encoding function - returns zeros for base policy
        
        Args:
            depth_image: Depth image (unused)
            proprio: Proprioceptive data
            
        Returns:
            depth_latent_yaw: Dummy depth features (zeros)
        """
        batch_size = proprio.shape[0]
        device = proprio.device
        
        # Return zeros for depth features (32) + yaw (2) = 34 dimensions
        dummy_depth_latent_yaw = torch.zeros(batch_size, 34, device=device, dtype=torch.float32)
        
        return dummy_depth_latent_yaw
    
    return encode_depth


def create_legged_loco_policy_function(policy):
    """
    Create policy function for legged-loco
    
    Args:
        policy: Loaded JIT policy
        
    Returns:
        actor_model: Policy function
    """
    def actor_model(obs):
        """
        Generate action using legged-loco policy
        
        Args:
            obs: Observation vector [batch_size, 450]
            
        Returns:
            action: Action vector [batch_size, 12]
        """
        action = policy(obs)
        return action
    
    return actor_model


@torch.inference_mode()
def main(args):
    rclpy.init()

    # 1. Load and parse configuration
    joint_map, default_joint_pos, kp, kd, obs_scales, action_scale, clip_obs, clip_actions, duration = load_configuration(args.logdir)
    device = "cuda"

    # 2. Create ROS node with configuration parameters
    handler = Go2Handler(
        joint_map=joint_map,
        default_joint_pos=default_joint_pos,
        device=device,
        dryrun=not args.nodryrun,
        mode=args.mode,
        kp=kp,
        kd=kd,
        action_scale=action_scale,
        clip_obs=clip_obs,
        clip_actions=clip_actions,
    )

    # 3. Print configuration information
    log_system_info(handler, args.logdir, duration)

    # 4. Load models and create inference functions
    turn_obs, encode_depth, actor_model = setup_models(args.logdir, device)

    # 5. Register models to node
    handler.register_models(turn_obs=turn_obs, depth_encode=encode_depth, policy=actor_model)
    handler.start_ros_handlers()
    handler.warm_up()

    # 6. Start control loop
    handle_timing_mode(handler, args.timing_mode, duration)

    # 7. Shutdown properly
    handler.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--logdir", type=str, default=None, help="The directory which contains the config.json and model_*.pt files")
    parser.add_argument("--nodryrun", action="store_true", default=False, help="Disable dryrun mode")
    parser.add_argument("--timing_mode", type=str, default="ros_timer",
        choices=["manual_control", "ros_timer"],
        help="Select timing mode: manual_control (precise timing control) or ros_timer (ROS managed timer)",
    )
    parser.add_argument("--mode", type=str, default="locomotion", choices=["locomotion", "walk"])
    args = parser.parse_args()
    
    main(args)
