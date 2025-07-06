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
            node.use_parkour_policy = False
    
    if node.use_stand_policy:
        stand_action = node.get_stand_action()
        node.send_stand_action(stand_action)
    
    if (node.joy_stick_buffer.keys & node.WirelessButtons.Y):
        node.get_logger().info("Y pressed, use the parkour policy")
        node.use_stand_policy = False
        node.use_parkour_policy = True
        node.use_sport_mode = False
        node.global_counter = 0

    if node.use_parkour_policy:
        node.use_stand_policy = False
        node.use_sport_mode = False
        
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
        node.get_logger().info("L2 pressed, stop using parkour policy, switch to sport mode.")
        node.use_stand_policy = False
        node.use_parkour_policy = False
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


@torch.inference_mode()
def main(args):
    rclpy.init()

    assert args.logdir is not None, "Please provide a logdir"
    with open(osp.join(args.logdir, "config.json"), "r") as f:
        config_dict = json.load(f, object_pairs_hook= OrderedDict)
    
    config_dict["control"]["computer_clip_torque"] = True
    
    # duration = config_dict["sim"]["dt"] * config_dict["control"]["decimation"] # different from parkour
    device = "cuda"
    duration = 0.02

    env_node = Go2ROS2Node(
        "go2",
        cfg= config_dict,
        model_device= device,
        dryrun= not args.nodryrun,
        mode = args.mode,
    )

    env_node.get_logger().info("Model loaded from: {}".format(osp.join(args.logdir)))
    env_node.get_logger().info("Control Duration: {} sec".format(duration))
    env_node.get_logger().info("Motor Stiffness (kp): {}".format(env_node.p_gains))
    env_node.get_logger().info("Motor Damping (kd): {}".format(env_node.d_gains))

    base_model_name = 'base_jit.pt'
    base_model_path = os.path.join(args.logdir, base_model_name)

    vision_model_name = 'vision_weight.pt'
    vision_model_path = os.path.join(args.logdir, vision_model_name)

    base_model = torch.jit.load(base_model_path, map_location=device)
    base_model.eval()

    estimator = base_model.estimator.estimator
    hist_encoder = base_model.actor.history_encoder
    actor = base_model.actor.actor_backbone

    vision_model = torch.load(vision_model_path, map_location=device)
    depth_backbone = DepthOnlyFCBackbone58x87(None, 32, 512)
    depth_encoder = RecurrentDepthBackbone(depth_backbone, None).to(device)
    depth_encoder.load_state_dict(vision_model['depth_encoder_state_dict'])
    depth_encoder.to(device)
    depth_encoder.eval()
    
    def turn_obs(proprio, depth_latent_yaw, proprio_history, n_proprio, n_depth_latent, n_hist_len):
        depth_latent = depth_latent_yaw[:, :-2]
        yaw = depth_latent_yaw[:, -2:] * 1.5
        print('yaw: ', yaw)
        
        proprio[:, 6:8] = yaw

        lin_vel_latent = estimator(proprio)

        activation = nn.ELU()
        priv_latent = hist_encoder(activation, proprio_history.view(-1, n_hist_len, n_proprio))

        
        obs = torch.cat([proprio, depth_latent, lin_vel_latent, priv_latent], dim=-1)

        return obs

    def encode_depth(depth_image, proprio):
        depth_latent_yaw = depth_encoder(depth_image, proprio)
        if torch.isnan(depth_latent_yaw).any():
            print('depth_latent_yaw contains nan and the depth image is: ', depth_image)
        return depth_latent_yaw
    
    def actor_model(obs):
        action = actor(obs)
        return action

    env_node.register_models(turn_obs=turn_obs, depth_encode=encode_depth, policy=actor_model)
    env_node.start_ros_handlers()
    env_node.warm_up()

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
    parser.add_argument("--mode", type= str, default= "parkour", choices=["parkour", "walk"])
    args = parser.parse_args()
    
    main(args)
