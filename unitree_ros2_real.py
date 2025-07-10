#!/usr/bin/env python3
import os, sys

import rclpy
from rclpy.node import Node
from unitree_go.msg import (
    WirelessController,
    LowState,
    SportModeState,
    LowCmd,
)
from unitree_api.msg import Request, RequestHeader

from std_msgs.msg import Float32MultiArray

if os.uname().machine in ["x86_64", "amd64"]:
    sys.path.append(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "x86",
    ))
elif os.uname().machine == "aarch64":
    sys.path.append(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "aarch64",
    ))
from crc_module import get_crc

from multiprocessing import Process
from collections import OrderedDict
import numpy as np
import torch
import time
import json


@torch.jit.script
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)

@torch.jit.script
def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = 1.0 - 2.0 * (q[:, qx] * q[:, qx] + q[:, qy] * q[:, qy])
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = 1.0 - 2.0 * (q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz])
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw
    
class UnitreeRos2Real(Node):
    """ A proxy implementation of the real Go2 robot. """
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

    def __init__(self,
            node_name= "unitree_ros2_real",
            robot_namespace= None,
            joy_stick_topic= "/wirelesscontroller",
            unitree_api_topic= "/api/sport/request",
            dryrun= True,
        ):
        super().__init__(node_name)
        self.robot_namespace = robot_namespace
        self.joy_stick_topic = joy_stick_topic
        self.unitree_api_topic = unitree_api_topic
        self.dryrun = dryrun

        # Use dictionaries for buffers to avoid attribute errors before messages are received
        self.joy_stick_buffer = {}
        self.sport_mode_state_buffer = {}

        self.start_ros_handlers()
        if not self.dryrun:
            self.sport_client = self.create_publisher(
                Request, self.unitree_api_topic, 10
            )

    def start_ros_handlers(self):
        """ start ros handlers for subscribers """
        self.joy_stick_sub = self.create_subscription(
            WirelessController,
            self.joy_stick_topic,
            self._joy_stick_callback,
            10,
        )

        self.sport_mode_state_sub = self.create_subscription(
            SportModeState,
            "/sportmodestate",
            self._sport_mode_state_callback,
            10,
        )

    def _joy_stick_callback(self, msg):
        self.joy_stick_buffer = msg

    def _sport_mode_state_callback(self, msg):
        self.sport_mode_state_buffer = msg
    
    def _sport_mode_change(self, mode, vel, yaw_speed=0.0, step_height=0.0):
        if self.dryrun:
            self.get_logger().info(f"Dryrun: Sport mode change request: mode={mode}, vel={vel}, yaw_speed={yaw_speed}, step_height={step_height}")
            return
        
        req = Request()
        # Set the API ID via the identity field in the header
        req.header.identity.api_id = mode
        req.header.identity.id = 0 # Not used, but good to initialize
        
        # Corrected field name from parameter_string to parameter
        # and adjusted JSON payload creation
        
        if mode == 1008: # MOVE
             sport_req = {
                "x": float(vel[0]),
                "y": float(vel[1]),
                "z": float(yaw_speed),
             }
             
             req.parameter = json.dumps(sport_req)
        else:
            # For other commands like stand, sit, etc., send an empty JSON object
            req.parameter = json.dumps({})
        # print(f"Sending sport command: {req}")
        self.sport_client.publish(req)


    def move(self, vx: float, vy: float, vyaw: float):
        """
        Commands the robot to move with specified linear and angular velocities.
        """
        # API ID for move is 1008
        self._sport_mode_change(1008, [vx, vy, 0.0], yaw_speed=vyaw)

    def stand(self):
        """
        Commands the robot to stand up.
        """
        # API ID for stand up is 1004
        self._sport_mode_change(1004, [0.0, 0.0, 0.0])

    def sit_down(self):
        """
        Commands the robot to sit down.
        """
        # API ID for stand down is 1005
        self._sport_mode_change(1005, [0.0, 0.0, 0.0])

    def balance_stand(self):
        """
        Commands the robot to a balance stand pose.
        """
        # API ID for balance stand is 1002
        self._sport_mode_change(1002, [0.0, 0.0, 0.0]) 