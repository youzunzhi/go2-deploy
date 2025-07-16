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


class Go2Runner:
    """Runner class for Go2 robot control system"""
    
    def __init__(self, args):
        """Initialize the Go2 runner with configuration and models"""
        self.args = args
        logdir = args.logdir
        device = args.device
        rclpy.init()
        
        # Load and parse configuration
        joint_map, default_joint_pos, kp, kd, action_scale, clip_obs, clip_actions, self.duration, self.policy_custom_manager = load_configuration(logdir, device)

        # Create ROS node with configuration parameters
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

        # Set handler to policy custom manager
        self.policy_custom_manager.set_handler(self.handler)

        # Create SportModeManager instance
        self.sport_mode_manager = SportModeManager(self.handler)
        
        # Print configuration information
        self.log_system_info()

        self.handler.start_ros_handlers()
        self.handler.warm_up()

    def main_loop(self):
        """Main control loop for the Go2 robot - handles different operational modes based on joystick input"""
        use_locomotion_policy = self.sport_mode_manager.sport_mode_before_locomotion()

        if use_locomotion_policy:
            action = self.policy_custom_manager.get_action()
            self.handler.send_action(action)
            self.handler.global_counter += 1

        if self.sport_mode_manager.sport_mode_after_locomotion():
            self.handler.log_info("L2 pressed, stop using locomotion policy, switch back to sport mode.")

    @torch.inference_mode()
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
    
    def log_system_info(self):
        """
        Print system configuration information
        
        Args:
            handler: Go2 Handler
            logdir: Model directory
            duration: Control cycle
        """
        self.handler.log_info("Model loaded from: {}".format(osp.join(self.args.logdir)))
        self.handler.log_info("Control Duration: {} sec".format(self.duration))
        self.handler.log_info("Motor Stiffness (kp): {}".format(self.handler.kp))
        self.handler.log_info("Motor Damping (kd): {}".format(self.handler.kd))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--logdir", type=str, default=None, help="The directory which contains the config.json and model_*.pt files")
    parser.add_argument("--nodryrun", action="store_true", default=False, help="Disable dryrun mode")
    parser.add_argument("--timing_mode", type=str, default="ros_timer",
        choices=["manual_control", "ros_timer"],
        help="Select timing mode: manual_control (precise timing control) or ros_timer (ROS managed timer)",
    )
    args = parser.parse_args()
    
    Go2Runner(args).run()
