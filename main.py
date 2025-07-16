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
from utils.sport_mode_manager import SportModeManager
from utils.obs import EPO_obs, LeggedLocoObs

class Go2Runner:
    """Runner class for Go2 robot control system"""
    
    def __init__(self, args):
        """Initialize the Go2 runner with configuration and models"""
        self.args = args
        rclpy.init()
        
        # 1. Load and parse configuration
        joint_map, default_joint_pos, kp, kd, obs_scales, action_scale, clip_obs, clip_actions, self.duration = load_configuration(args.logdir)
        device = "cuda"

        # 2. Create ROS node with configuration parameters
        self.handler = Go2Handler(
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

        # Create SportModeManager instance
        self.sport_mode_manager = SportModeManager(self.handler)

        self.obs_manager = EPO_obs(
        
        # 3. Print configuration information
        log_system_info(self.handler, args.logdir, self.duration)

        # 4. Load models and create inference functions
        turn_obs, encode_depth, actor_model = setup_models(args.logdir, device)

        # 5. Register models to node
        self.handler.register_models(turn_obs=turn_obs, depth_encode=encode_depth, policy=actor_model)
        self.handler.start_ros_handlers()
        self.handler.warm_up()
    
    def run(self):
        """Run the main control loop"""
        try:
            # Start control loop
            self._handle_timing_mode()
        finally:
            # Shutdown properly
            self.handler.shutdown()
            rclpy.shutdown()
    
    def _start_main_loop_timer(self):
        """Start the main loop timer for ROS-based timing control"""
        self.handler.main_loop_timer = self.handler.node.create_timer(
            self.duration, # in sec
            self.main_loop,
        )
    
    def _handle_timing_mode(self):
        """Handle different timing modes for the control loop"""
        if self.args.timing_mode == "ros_timer":
            # Use ROS timer for timing control
            self.handler.log_info('Model and Policy are ready')
            self._start_main_loop_timer()
            rclpy.spin(self.handler.node)
        
        elif self.args.timing_mode == "manual_control":
            # Manually control timing for more precise control
            rclpy.spin_once(self.handler.node, timeout_sec=0.)
            self.handler.log_info("Model and Policy are ready")
            
            while rclpy.ok():
                # Track iteration time to maintain desired frequency
                main_loop_time = time.monotonic()
                
                # Run one iteration
                self.main_loop()
                rclpy.spin_once(self.handler.node, timeout_sec=0.)
                
                # Sleep remaining time to maintain frequency
                sleep_time = max(0, self.duration - (time.monotonic() - main_loop_time))
                time.sleep(sleep_time)
        
        else:
            raise ValueError(f"Invalid timing mode: {self.args.timing_mode}")
    
    def main_loop(self):
        """Main control loop for the Go2 robot - handles different operational modes based on joystick input"""
        use_locomotion_policy = self.sport_mode_manager.sport_mode_before_locomotion()

        if use_locomotion_policy:
            obs = 
            action = self.handler.policy(obs)
            self.handler.send_action(action)
            self.handler.global_counter += 1

        if self.sport_mode_manager.sport_mode_after_locomotion():
            self.handler.log_info("L2 pressed, stop using locomotion policy, switch back to sport mode.")

def log_system_info(handler, logdir, duration):
    """
    Print system configuration information
    
    Args:
        handler: Go2 Handler
        logdir: Model directory
        duration: Control cycle
    """
    handler.log_info("Model loaded from: {}".format(osp.join(logdir)))
    handler.log_info("Control Duration: {} sec".format(duration))
    handler.log_info("Motor Stiffness (kp): {}".format(handler.kp))
    handler.log_info("Motor Damping (kd): {}".format(handler.kd))


# def setup_models(logdir, device):
#     """
#     Unified setup of models and inference functions
    
#     Args:
#         logdir: Model file directory
#         device: Computing device
        
#     Returns:
#         turn_obs: Observation processing function
#         encode_depth: Depth encoding function
#         actor_model: Policy function
#     """
#     # Determine policy source from logdir structure
#     policy_source = determine_policy_source(logdir)
    
#     if policy_source == "EPO":
#         # EPO: Load separate base model and vision model
#         base_model, estimator, hist_encoder, actor = load_base_model(logdir, device)
#         depth_encoder = load_vision_model(logdir, device)
        
#         # Create inference functions for EPO
#         turn_obs = create_observation_processor(estimator, hist_encoder)
#         encode_depth = create_depth_encoder(depth_encoder)
#         actor_model = create_policy(actor)
        
#     elif policy_source == "legged-loco":
#         # legged-loco: Load single JIT policy
#         policy = load_legged_loco_policy(logdir, device)
        
#         # Create inference functions for legged-loco
#         turn_obs = create_legged_loco_observation_processor()
#         encode_depth = create_dummy_depth_encoder()  # No vision in base policy
#         actor_model = create_legged_loco_policy_function(policy)
        
#     else:
#         raise ValueError(f"Unsupported policy source: {policy_source}")
    
#     return turn_obs, encode_depth, actor_model


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


@torch.inference_mode()
def main(args):
    """Main entry point using Go2Runner class"""
    runner = Go2Runner(args)
    runner.run()


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
