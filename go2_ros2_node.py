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
from crc_module import get_crc  # type: ignore

from multiprocessing import Process
from collections import OrderedDict
import numpy as np
import torch
import time


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

class Go2RobotCfgs:
    NUM_DOF = 12
    NUM_ACTIONS = 12
    
    @staticmethod
    def get_config_for_policy_source(policy_source="EPO"):
        """Get robot configuration based on policy source to handle different joint orderings"""
        config = {
            'NUM_DOF': 12,
            'NUM_ACTIONS': 12,
            'dof_signs': [1.] * 12,
            'turn_on_motor_mode': [0x01] * 12,
        }
        
        if policy_source == "legged-loco":
            # legged-loco uses grouped joint order: [all hips, all thighs, all calves]
            # Simulation order: FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, FR_thigh, RL_thigh, RR_thigh, FL_calf, FR_calf, RL_calf, RR_calf
            config['dof_names'] = [
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
            # Mapping from legged-loco simulation order to hardware order
            # Hardware order: FR_hip(0), FR_thigh(1), FR_calf(2), FL_hip(3), FL_thigh(4), FL_calf(5), RR_hip(6), RR_thigh(7), RR_calf(8), RL_hip(9), RL_thigh(10), RL_calf(11)
            # legged-loco sim: FL_hip(0), FR_hip(1), RL_hip(2), RR_hip(3), FL_thigh(4), FR_thigh(5), RL_thigh(6), RR_thigh(7), FL_calf(8), FR_calf(9), RL_calf(10), RR_calf(11)
            config['dof_map'] = [
                3, 0, 9, 6,   # Hip joints: FL->3, FR->0, RL->9, RR->6  
                4, 1, 10, 7,  # Thigh joints: FL->4, FR->1, RL->10, RR->7
                5, 2, 11, 8   # Calf joints: FL->5, FR->2, RL->11, RR->8
            ]
            # Joint limits in legged-loco simulation order
            config['joint_limits_high'] = torch.tensor([
                1.0472, 1.0472, 1.0472, 1.0472,    # Hip joints: FL, FR, RL, RR
                3.4907, 3.4907, 4.5379, 4.5379,    # Thigh joints: FL, FR, RL, RR  
                -0.83776, -0.83776, -2.7227, -2.7227  # Calf joints: FL, FR, RL, RR
            ], device="cpu", dtype=torch.float32)
            config['joint_limits_low'] = torch.tensor([
                -1.0472, -1.0472, -1.0472, -1.0472,  # Hip joints: FL, FR, RL, RR
                -1.5708, -1.5708, -0.5236, -0.5236,  # Thigh joints: FL, FR, RL, RR
                -2.7227, -2.7227, -2.7227, -2.7227   # Calf joints: FL, FR, RL, RR
            ], device="cpu", dtype=torch.float32)
            config['torque_limits'] = torch.tensor([
                25, 25, 25, 25,  # Hip joints: FL, FR, RL, RR
                40, 40, 40, 40,  # Thigh joints: FL, FR, RL, RR
                40, 40, 40, 40   # Calf joints: FL, FR, RL, RR
            ], device="cpu", dtype=torch.float32)
        else:
            # EPO/default uses leg-grouped joint order: [FR leg, FL leg, RR leg, RL leg]
            # Simulation order matches hardware order (reindexed in simulation)
            config['dof_names'] = [
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
            # Identity mapping (simulation order already matches hardware)
            config['dof_map'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            # Joint limits in EPO simulation order (leg-grouped)
            config['joint_limits_high'] = torch.tensor([
                1.0472, 3.4907, -0.83776,    # FR leg
                1.0472, 3.4907, -0.83776,    # FL leg
                1.0472, 4.5379, -0.83776,    # RR leg
                1.0472, 4.5379, -0.83776,    # RL leg
            ], device="cpu", dtype=torch.float32)
            config['joint_limits_low'] = torch.tensor([
                -1.0472, -1.5708, -2.7227,   # FR leg
                -1.0472, -1.5708, -2.7227,   # FL leg
                -1.0472, -0.5236, -2.7227,   # RR leg
                -1.0472, -0.5236, -2.7227,   # RL leg
            ], device="cpu", dtype=torch.float32)
            config['torque_limits'] = torch.tensor([
                25, 40, 40,  # FR leg
                25, 40, 40,  # FL leg
                25, 40, 40,  # RR leg
                25, 40, 40,  # RL leg
            ], device="cpu", dtype=torch.float32)
        
        return config
    
    

class Go2ROS2Node(Node):
    """ A proxy implementation of the real H1 robot. """
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
            robot_namespace= None,
            low_state_topic= "/lowstate",
            low_cmd_topic= "/lowcmd",
            joy_stick_topic= "/wirelesscontroller",
            depth_data_topic= "/forward_depth_image",
            # Configuration parameters that need to be passed in
            clip_obs=100,
            clip_actions=None,
            obs_scales=None,
            control_type=None,
            kp=40.,
            kd=1.,
            action_scale= None,
            default_joint_angles= None,
            lin_vel_deadband= 0.1,
            ang_vel_deadband= 0.1,
            cmd_px_range= [0.4, 1.0], # check joy_stick_callback (p for positive, n for negative)
            cmd_nx_range= [0.4, 0.8], # check joy_stick_callback (p for positive, n for negative)
            cmd_py_range= [0.4, 0.8], # check joy_stick_callback (p for positive, n for negative)
            cmd_ny_range= [0.4, 0.8], # check joy_stick_callback (p for positive, n for negative)
            cmd_pyaw_range= [0.4, 1.6], # check joy_stick_callback (p for positive, n for negative)
            cmd_nyaw_range= [0.4, 1.6], # check joy_stick_callback (p for positive, n for negative)
            move_by_wireless_remote= True, # if True, the robot will be controlled by a wireless remote
            model_device= "cpu",
            dof_pos_protect_ratio= 1.1, # if the dof_pos is out of the range of this ratio, the process will shutdown.
            robot_class_name= "Go2",
            dryrun= True, # if True, the robot will not send commands to the real robot
            mode= "locomotion",
            policy_source= "EPO", # Policy source: "EPO" or "legged-loco"
        ):
        super().__init__("unitree_ros2_real")
        self.robot_namespace = robot_namespace
        self.low_state_topic = low_state_topic
        self.low_cmd_topic = low_cmd_topic if not dryrun else low_cmd_topic + "_dryrun_" + str(np.random.randint(0, 65535))
        self.joy_stick_topic = joy_stick_topic
        self.depth_data_topic = depth_data_topic
        
        # Store configuration parameters directly as attributes
        self.clip_obs = clip_obs
        self.clip_actions = clip_actions
        self.obs_scales = obs_scales
        self.control_type = control_type
        self.kp = kp
        self.kd = kd
        self.action_scale = action_scale
        self.default_joint_angles = default_joint_angles
        
        self.lin_vel_deadband = lin_vel_deadband
        self.ang_vel_deadband = ang_vel_deadband
        self.cmd_px_range = cmd_px_range
        self.cmd_nx_range = cmd_nx_range
        self.cmd_py_range = cmd_py_range
        self.cmd_ny_range = cmd_ny_range
        self.cmd_pyaw_range = cmd_pyaw_range
        self.cmd_nyaw_range = cmd_nyaw_range
        self.move_by_wireless_remote = move_by_wireless_remote
        self.model_device = model_device
        self.dof_pos_protect_ratio = dof_pos_protect_ratio
        self.robot_class_name = robot_class_name
        self.dryrun = dryrun
        self.mode = mode
        self.policy_source = policy_source

        # Get policy-specific robot configuration
        self.robot_config = Go2RobotCfgs.get_config_for_policy_source(policy_source)
        self.NUM_DOF = self.robot_config['NUM_DOF']
        self.NUM_ACTIONS = self.robot_config['NUM_ACTIONS']
        self.dof_map = self.robot_config['dof_map']
        self.dof_names = self.robot_config['dof_names']
        self.dof_signs = self.robot_config['dof_signs']
        self.turn_on_motor_mode = self.robot_config['turn_on_motor_mode']

        # Set observation dimensions based on policy source
        if policy_source == "legged-loco":
            self.n_proprio = 45  # base_ang_vel(3) + base_rpy(3) + velocity_commands(3) + joint_pos(12) + joint_vel(12) + actions(12)
        else:
            self.n_proprio = 53  # EPO format: ang_vel(3) + imu(2) + yaw_info(3) + commands(3) + locomotion_walk(2) + dof_pos(12) + dof_vel(12) + last_actions(12) + contact(4)
        self.n_depth_latent = 32
        self.n_hist_len = 10

        self.proprio_history_buf = torch.zeros(1, self.n_hist_len, self.n_proprio, device=self.model_device, dtype=torch.float)
        self.episode_length_buf = torch.zeros(1, device=self.model_device, dtype=torch.float)
        self.forward_depth_latent_yaw_buffer = torch.zeros(1, self.n_depth_latent+2, device=self.model_device, dtype=torch.float)
        self.xyyaw_command = torch.tensor([[0, 0, 0]], device= self.model_device, dtype= torch.float32)
        self.contact_filt = torch.ones((1, 4), device= self.model_device, dtype= torch.float32)
        self.last_contact_filt = torch.ones((1, 4), device= self.model_device, dtype= torch.float32)

        self.setup_control_gains_and_buffers()
        self.init_stand_config()
        
        self.global_counter = 0
        self.visual_update_interval = 5
        self.use_stand_policy = False
        self.use_locomotion_policy = False
        self.use_sport_mode = True

    def init_stand_config(self):
        self.startPos = [0.0] * 12
        self._targetPos_1 = [0.0, 1.36, -2.65, 0.0, 1.36, -2.65,
                             -0.2, 1.36, -2.65, 0.2, 1.36, -2.65]
        self._targetPos_2 = [0.0, 0.67, -1.3, 0.0, 0.67, -1.3,
                             0.0, 0.67, -1.3, 0.0, 0.67, -1.3]
        self.stand_action = [0.0] * 12

        self.duration_1 = 10
        self.duration_2 = 100
        self.percent_1 = 0
        self.percent_2 = 0

        self.firstrun_target_1 = True
        self.firstRun = True

    def reset_obs(self):
        self.startPos = [0.0] * 12
        self.stand_action = [0.0] * 12

        self.percent_1 = 0
        self.percent_2 = 0

        self.firstrun_target_1 = True
        self.firstRun = True

        self.actions = torch.zeros(self.NUM_ACTIONS, device= self.model_device, dtype= torch.float32)    
        self.proprio_history_buf = torch.zeros(1, self.n_hist_len, self.n_proprio, device=self.model_device, dtype=torch.float)
        self.episode_length_buf = torch.zeros(1, device=self.model_device, dtype=torch.float)
        self.forward_depth_latent_yaw_buffer = torch.zeros(1, self.n_depth_latent+2, device=self.model_device, dtype=torch.float)
        self.xyyaw_command = torch.tensor([[0, 0, 0]], device= self.model_device, dtype= torch.float32)
        self.contact_filt = torch.ones((1, 4), device= self.model_device, dtype= torch.float32)
        self.last_contact_filt = torch.ones((1, 4), device= self.model_device, dtype= torch.float32)


    def setup_control_gains_and_buffers(self):
        """ Setup default positions, and buffers from the config parameters """

        self.default_joint_angles = torch.zeros(self.NUM_DOF, device= self.model_device, dtype= torch.float32)
        self.dof_pos_ = torch.empty(1, self.NUM_DOF, device= self.model_device, dtype= torch.float32)
        self.dof_vel_ = torch.empty(1, self.NUM_DOF, device= self.model_device, dtype= torch.float32)
        
        for i in range(self.NUM_DOF):
            name = self.dof_names[i]
            default_joint_angle = self.default_joint_angles[name]
            self.default_joint_angles[i] = default_joint_angle

        # actions
        self.num_actions = self.NUM_ACTIONS
        self.get_logger().info("[Env] action scale: {:.2f}".format(self.action_scale))
        
        self.actions = torch.zeros(self.NUM_ACTIONS, device= self.model_device, dtype= torch.float32)    

        ###################### hardware related #####################
        # Use policy-specific joint and torque limits
        self.joint_limits_high = self.robot_config['joint_limits_high'].to(self.model_device)
        self.joint_limits_low = self.robot_config['joint_limits_low'].to(self.model_device)
        self.torque_limits = self.robot_config['torque_limits'].to(self.model_device)
        joint_pos_mid = (self.joint_limits_high + self.joint_limits_low) / 2
        joint_pos_range = (self.joint_limits_high - self.joint_limits_low) / 2
        self.joint_pos_protect_high = joint_pos_mid + joint_pos_range * self.dof_pos_protect_ratio
        self.joint_pos_protect_low = joint_pos_mid - joint_pos_range * self.dof_pos_protect_ratio

    def start_ros_handlers(self):
        """ after initializing the env and policy, register ros related callbacks and topics
        """
        # ROS publishers
        self.low_cmd_pub = self.create_publisher(
            LowCmd,
            self.low_cmd_topic,
            1
        )
        self.low_cmd_buffer = LowCmd()

        # ROS subscribers
        self.low_state_sub = self.create_subscription(
            LowState,
            self.low_state_topic,
            self._low_state_callback,
            1
        )
        self.get_logger().info("Low state subscriber started, waiting to receive low state messages.")

        self.joy_stick_sub = self.create_subscription(
            WirelessController,
            self.joy_stick_topic,
            self._joy_stick_callback,
            1
        )
        self.get_logger().info("Wireless controller subscriber started, waiting to receive wireless controller messages.")

        self.sport_state_pub = self.create_publisher(
            Request,
            '/api/robot_state/request',
            1,
        )

        self.sport_mode_pub = self.create_publisher(
            Request,
            '/api/sport/request',
            1,
        )

        self.motion_switcher_pub = self.create_publisher(
            Request,
            '/api/motion_switcher/request',
            1,
        )

        self.depth_input_sub = self.create_subscription(
            Float32MultiArray,
            self.depth_data_topic,
            self._depth_data_callback,
            1
        )

        self.get_logger().info("ROS handlers started, waiting to recieve critical low state and wireless controller messages.")
        if not self.dryrun:
            self.get_logger().warn(f"You are running the code in no-dryrun mode and publishing to '{self.low_cmd_topic}', Please keep safe.")
        else:
            self.get_logger().warn(f"You are publishing low cmd to '{self.low_cmd_topic}' because of dryrun mode, Please check and be safe.")
        while rclpy.ok():
            rclpy.spin_once(self)
            if hasattr(self, "low_state_buffer") and hasattr(self, "joy_stick_buffer"):
                break
        self.get_logger().info("Low state and wireless message received, the robot is ready to go.")

    """ ROS callbacks and handlers that update the buffer """

    def _low_state_callback(self, msg):
        # self.get_logger().warn("Low state message received.")
        """ store and handle proprioception data """
        self.low_state_buffer = msg # keep the latest low state

        ################### refresh dof_pos and dof_vel ######################
        for sim_idx in range(self.NUM_DOF):
            real_idx = self.dof_map[sim_idx]
            self.dof_pos_[0, sim_idx] = self.low_state_buffer.motor_state[real_idx].q * self.dof_signs[sim_idx]
        for sim_idx in range(self.NUM_DOF):
            real_idx = self.dof_map[sim_idx]
            self.dof_vel_[0, sim_idx] = self.low_state_buffer.motor_state[real_idx].dq * self.dof_signs[sim_idx]

    def _joy_stick_callback(self, msg):
        # self.get_logger().warn("Wireless controller message received.")
        self.joy_stick_buffer = msg
        if self.move_by_wireless_remote:
            # left-y for forward/backward
            ly = msg.ly
            if ly > self.lin_vel_deadband:
                vx = (ly - self.lin_vel_deadband) / (1 - self.lin_vel_deadband) # (0, 1)
                vx = vx * (self.cmd_px_range[1] - self.cmd_px_range[0]) + self.cmd_px_range[0]
            elif ly < -self.lin_vel_deadband:
                vx = (ly + self.lin_vel_deadband) / (1 - self.lin_vel_deadband) # (-1, 0)
                vx = vx * (self.cmd_nx_range[1] - self.cmd_nx_range[0]) - self.cmd_nx_range[0]
            else:
                vx = 0
            # left-x for turning left/right
            lx = -msg.lx
            if lx > self.ang_vel_deadband:
                yaw = (lx - self.ang_vel_deadband) / (1 - self.ang_vel_deadband)
                yaw = yaw * (self.cmd_pyaw_range[1] - self.cmd_pyaw_range[0]) + self.cmd_pyaw_range[0]
            elif lx < -self.ang_vel_deadband:
                yaw = (lx + self.ang_vel_deadband) / (1 - self.ang_vel_deadband)
                yaw = yaw * (self.cmd_nyaw_range[1] - self.cmd_nyaw_range[0]) - self.cmd_nyaw_range[0]
            else:
                yaw = 0
            # right-x for side moving left/right
            rx = -msg.rx
            if rx > self.lin_vel_deadband:
                vy = (rx - self.lin_vel_deadband) / (1 - self.lin_vel_deadband)
                vy = vy * (self.cmd_py_range[1] - self.cmd_py_range[0]) + self.cmd_py_range[0]
            elif rx < -self.lin_vel_deadband:
                vy = (rx + self.lin_vel_deadband) / (1 - self.lin_vel_deadband)
                vy = vy * (self.cmd_ny_range[1] - self.cmd_ny_range[0]) - self.cmd_ny_range[0]
            else:
                vy = 0
            self.xyyaw_command = torch.tensor([[vx, vy, yaw]], device= self.model_device, dtype= torch.float32)

        # refer to Unitree Remote Control data structure, msg.keys is a bit mask
        # 00000000 00000001 means pressing the 0-th button (R1)
        # 00000000 00000010 means pressing the 1-th button (L1)
        # 10000000 00000000 means pressing the 15-th button (left)
        
        # if (msg.keys & self.WirelessButtons.R2) or (msg.keys & self.WirelessButtons.L2): # R2 or L2 is pressed
        # if  msg.keys & self.WirelessButtons.L2: # R2 or L2 is pressed
        #     self.get_logger().warn("R2 or L2 is pressed, the motors and this process shuts down.")
        #     self._turn_off_motors()
        #     raise SystemExit()

        # roll-pitch target
        if hasattr(self, "roll_pitch_yaw_cmd"):
            if (msg.keys & self.WirelessButtons.up):
                self.roll_pitch_yaw_cmd[0, 1] += 0.1
                self.get_logger().info("Pitch Command: " + str(self.roll_pitch_yaw_cmd))
            if (msg.keys & self.WirelessButtons.down):
                self.roll_pitch_yaw_cmd[0, 1] -= 0.1
                self.get_logger().info("Pitch Command: " + str(self.roll_pitch_yaw_cmd))
            if (msg.keys & self.WirelessButtons.left):
                self.roll_pitch_yaw_cmd[0, 0] -= 0.1
                self.get_logger().info("Roll Command: " + str(self.roll_pitch_yaw_cmd))
            if (msg.keys & self.WirelessButtons.right):
                self.roll_pitch_yaw_cmd[0, 0] += 0.1
                self.get_logger().info("Roll Command: " + str(self.roll_pitch_yaw_cmd))

    def _depth_data_callback(self, msg):
        self.depth_data = torch.tensor(msg.data, dtype=torch.float32).reshape(1, 58, 87).to(self.model_device)

    
    def _sport_mode_change(self, mode):
        msg = Request()

        msg.header.identity.id = 0
        msg.header.identity.api_id = mode
        msg.header.lease.id = 0
        msg.header.policy.priority = 0
        msg.header.policy.noreply = False

        msg.parameter = ''
        msg.binary = []

        self.sport_mode_pub.publish(msg)
    
    def _sport_state_change(self, mode):
        msg = Request()

        # Fill the header
        msg.header.identity.id = 0
        msg.header.lease.id = 0
        msg.header.policy.priority = 0
        msg.header.policy.noreply = False

        if mode == 0:
            # Release mode (switch to normal mode) - use api_id 1003
            msg.header.identity.api_id = 1003
            msg.parameter = '{}'
        elif mode == 1:
            # Select MCF mode - use api_id 1002
            msg.header.identity.api_id = 1002
            msg.parameter = '{"name": "mcf"}'
        
        msg.binary = []

        # Publish to motion switcher instead of robot state
        self.motion_switcher_pub.publish(msg)

    """ Done: ROS callbacks and handlers that update the buffer """

    """ refresh observation buffer and corresponding sub-functions """
    
    def _get_ang_vel_obs(self):
        ang_vel = torch.from_numpy(self.low_state_buffer.imu_state.gyroscope).unsqueeze(0).to(device=self.model_device, dtype=torch.float32)
        return ang_vel * self.obs_scales["ang_vel"]
    
    def _get_imu_obs(self):
        quat_xyzw = torch.tensor([
            self.low_state_buffer.imu_state.quaternion[1],
            self.low_state_buffer.imu_state.quaternion[2],
            self.low_state_buffer.imu_state.quaternion[3],
            self.low_state_buffer.imu_state.quaternion[0],
            ], device= self.model_device, dtype= torch.float32).unsqueeze(0)
        roll, pitch, yaw = get_euler_xyz(quat_xyzw)
        imu_obs = torch.tensor([[roll, pitch]], device= self.model_device, dtype= torch.float32)
        return imu_obs

    def _get_base_rpy_obs(self):
        """Get base roll-pitch-yaw for legged-loco policy"""
        quat_xyzw = torch.tensor([
            self.low_state_buffer.imu_state.quaternion[1],
            self.low_state_buffer.imu_state.quaternion[2],
            self.low_state_buffer.imu_state.quaternion[3],
            self.low_state_buffer.imu_state.quaternion[0],
            ], device= self.model_device, dtype= torch.float32).unsqueeze(0)
        roll, pitch, yaw = get_euler_xyz(quat_xyzw)
        base_rpy = torch.tensor([[roll, pitch, yaw]], device= self.model_device, dtype= torch.float32)
        return base_rpy

    def _get_delta_yaw_obs(self):
        yaw = 0
        delta_yaw, delta_next_yaw = 0, 0
        yaw_info = torch.tensor([[0, delta_yaw, delta_next_yaw]], device= self.model_device, dtype= torch.float32)
        return yaw_info

    def _get_commands_obs(self):
        if self.move_by_wireless_remote:
            vx, vy, yaw = self.xyyaw_command[0, :]
            if self.policy_source == "legged-loco":
                # legged-loco uses full 3DOF commands: [lin_vel_x, lin_vel_y, ang_vel_z]
                commands = torch.tensor([[vx, vy, yaw]], device= self.model_device, dtype= torch.float32)
            else:
                # EPO uses only forward velocity: [0, 0, vx]
                commands = torch.tensor([[0, 0, vx]], device= self.model_device, dtype= torch.float32)
            return commands
        else:
            return torch.tensor([[0., 0., 0.]], device=self.model_device)

    def _get_dof_pos_obs(self):
        return (self.dof_pos_ - self.default_joint_angles.unsqueeze(0)) * self.obs_scales["dof_pos"]

    def _get_dof_vel_obs(self):
        return self.dof_vel_ * self.obs_scales["dof_vel"]

    def _get_last_actions_obs(self):
        return self.actions

    def _get_contact_filt_obs(self):
        for i in range(4):
            if self.low_state_buffer.foot_force[i] < 25:
                self.contact_filt[:, i] = -0.5
            else:
                self.contact_filt[:, i] = 0.5
        return self.contact_filt

    def _get_depth_image(self):
        return self.depth_data

    def _get_history_proprio(self):
        return self.proprio_history_buf
    

    def get_proprio(self):
        """ Observation segment is defined as a list of lists/ints defining the tensor shape with
        corresponding order.
        """
        start_time = time.monotonic()

        ang_vel = self._get_ang_vel_obs()  # (1, 3)
        ang_vel_time = time.monotonic()

        commands = self._get_commands_obs()  # (1, 3)
        commands_time = time.monotonic()

        dof_pos = self._get_dof_pos_obs()  # (1, 12)
        dof_pos_time = time.monotonic()

        dof_vel = self._get_dof_vel_obs()  # (1, 12)
        dof_vel_time = time.monotonic()

        last_actions = self._get_last_actions_obs().view(1, -1)  # (1, 12)
        last_action_time = time.monotonic()

        # Check if using legged-loco policy format
        if hasattr(self, 'policy_source') and self.policy_source == 'legged-loco':
            # legged-loco format: base_ang_vel, base_rpy, velocity_commands, joint_pos, joint_vel, actions
            base_rpy = self._get_base_rpy_obs()  # (1, 3)
            
            proprio = torch.cat([ang_vel, base_rpy, commands,
                            dof_pos, dof_vel, last_actions], dim=-1)
        else:
            # EPO format: ang_vel, imu, yaw_info, commands, locomotion_walk, dof_pos, dof_vel, last_actions, contact
            imu = self._get_imu_obs()  # (1, 2)
            imu_time = time.monotonic()

            yaw_info = self._get_delta_yaw_obs()  # (1, 3)
            yaw_time = time.monotonic()

            if self.mode == "locomotion":
                locomotion_walk = torch.tensor([[1, 0]], device= self.model_device, dtype= torch.float32) # locomotion
            elif self.mode == "walk":
                locomotion_walk = torch.tensor([[0, 1]], device= self.model_device, dtype= torch.float32) # walk

            contact = self._get_contact_filt_obs()  # (1, 4)
            contact_time = time.monotonic()
            
            proprio = torch.cat([ang_vel, imu, yaw_info, commands, locomotion_walk,
                            dof_pos, dof_vel, last_actions, contact], dim=-1)

        self.proprio_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([proprio] * self.n_hist_len, dim=1),
            torch.cat([
                self.proprio_history_buf[:, 1:],
                proprio.unsqueeze(1)
            ], dim=1)
        )
        end_time = time.monotonic()

        # print('ang vel time: {:.5f}'.format(ang_vel_time - start_time),
        #         'imu time: {:.5f}'.format(imu_time - ang_vel_time),
        #         'yaw time: {:.5f}'.format(yaw_time - imu_time),
        #         'command time: {:.5f}'.format(commands_time - yaw_time),
        #         'dof pos time: {:.5f}'.format(dof_pos_time - commands_time),
        #         'dof vel time: {:.5f}'.format(dof_vel_time - dof_pos_time),
        #         'last action time: {:.5f}'.format(last_action_time - dof_vel_time),
        #         'contact time: {:.5f}'.format(contact_time - last_action_time)
        #         )
        
        self.episode_length_buf += 1

        return proprio


    def clip_actions_by_joint_limits(self, robot_coordinates_action):
        """
        Clip actions to ensure joint positions stay within safe limits
        
        Args:
            robot_coordinates_action: Actions in robot coordinate system (batch_size, NUM_DOF)
            
        Returns:
            clipped_action: Actions clipped to joint limits
        """
        # Use instance joint limits (policy-specific)
        clipped_action = torch.clamp(robot_coordinates_action, 
                                   self.joint_limits_low.unsqueeze(0), 
                                   self.joint_limits_high.unsqueeze(0))
        
        return clipped_action
    
    def clip_actions_by_torque_limits(self, robot_coordinates_action, current_joint_pos, current_joint_vel):
        """
        Clip actions to ensure torque output stays within safe limits
        
        Args:
            robot_coordinates_action: Target positions in robot coordinate system (batch_size, NUM_DOF)
            current_joint_pos: Current joint positions (batch_size, NUM_DOF)
            current_joint_vel: Current joint velocities (batch_size, NUM_DOF)
            
        Returns:
            clipped_action: Actions clipped to torque limits
        """
        # Convert torque limits to position limits using PD control formula
        # tau = kp * (target - current) - kd * vel
        # For |tau| <= torque_limit:
        # target_min = current + (-torque_limit + kd * vel) / kp
        # target_max = current + (torque_limit + kd * vel) / kp
        torque_limit = self.torque_limits.unsqueeze(0)  # (1, NUM_DOF)
        
        # Calculate position limits based on torque constraints
        target_min = current_joint_pos + (-torque_limit + self.kd * current_joint_vel) / self.kp
        target_max = current_joint_pos + (torque_limit + self.kd * current_joint_vel) / self.kp
        
        # Clip robot coordinates action to these limits
        clipped_action = torch.clamp(robot_coordinates_action, target_min, target_max)
        
        return clipped_action

    def send_action(self, actions):
        """ Send the action to the robot motors, which does the preprocessing
        just like env.step in simulation.
        Thus, the actions has the batch dimension, whose size is 1.
        """
        if isinstance(actions, list):
            actions = torch.tensor(actions, device=self.model_device).unsqueeze(0)
        
        self.actions = actions

        # Check if clip_actions exists
        if self.clip_actions is not None:
            # EPO clipping
            hard_clip = self.clip_actions / self.action_scale
            clipped_scaled_action = torch.clip(actions, -hard_clip, hard_clip) * self.action_scale
        else:
            # legged-loco (no clipping)
            clipped_scaled_action = actions * self.action_scale

        robot_coordinates_action = clipped_scaled_action + self.default_joint_angles.unsqueeze(0)
        
        # Apply safety clipping using joint limits
        robot_coordinates_action = self.clip_actions_by_joint_limits(robot_coordinates_action)
        
        # Apply safety clipping using torque limits
        current_joint_pos = self.dof_pos_
        current_joint_vel = self.dof_vel_
        robot_coordinates_action = self.clip_actions_by_torque_limits(robot_coordinates_action, current_joint_pos, current_joint_vel)
        
        self._publish_legs_cmd(robot_coordinates_action[0], stand=False)

    def send_stand_action(self, actions):
        """ Send the action to the robot motors, which does the preprocessing
        just like env.step in simulation.
        Thus, the actions has the batch dimension, whose size is 1.
        """
        actions = torch.tensor(actions, device=self.model_device).unsqueeze(0)
        self.actions = actions

        self._publish_legs_cmd(actions[0], stand=True)

    def get_stand_action(self):
        if self.firstRun:
            for i in range(12):
                self.startPos[i] = self.low_state_buffer.motor_state[i].q
            self.firstRun = False

        self.percent_1 += 1.0 / self.duration_1
        self.percent_1 = min(self.percent_1, 1)
        if self.percent_1 < 1:
            for i in range(12):
                self.stand_action[i] = (1 - self.percent_1) * self.startPos[i] + self.percent_1 * self._targetPos_1[i]

            if self.firstrun_target_1:
                self.get_logger().info('Going to target Pos 1.', once=True)
                self.firstrun_target_1 = False
                self.firstrun_target_2 = True
        if (self.percent_1 == 1) and (self.percent_2 <= 1):
            self.percent_2 += 1.0 / self.duration_2
            self.percent_2 = min(self.percent_2, 1)
            for i in range(12):
                self.stand_action[i] = (1 - self.percent_2) * self._targetPos_1[i] + self.percent_2 * self._targetPos_2[i]

            self.get_logger().info('Staying in target Pos 2.', once=True)

        return self.stand_action

    """ functions that actually publish the commands and take effect """
    def _publish_legs_cmd(self, robot_coordinates_action, stand):
        """ Publish the joint commands to the robot legs in robot coordinates system.
        robot_coordinates_action: shape (NUM_DOF,), in simulation order.
        """
        #################### check ##############################
        for sim_idx in range(self.NUM_DOF):
            real_idx = self.dof_map[sim_idx]
            if not self.dryrun:
                self.low_cmd_buffer.motor_cmd[real_idx].mode = self.turn_on_motor_mode[sim_idx]
            self.low_cmd_buffer.motor_cmd[real_idx].q = robot_coordinates_action[sim_idx].item() * self.dof_signs[sim_idx]
            self.low_cmd_buffer.motor_cmd[real_idx].dq = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].tau = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].kp = self.kp
            self.low_cmd_buffer.motor_cmd[real_idx].kd = self.kd
        
        self.low_cmd_buffer.crc = get_crc(self.low_cmd_buffer)
        self.low_cmd_pub.publish(self.low_cmd_buffer)

    def _turn_off_motors(self):
        """ Turn off the motors """
        for sim_idx in range(self.NUM_DOF):
            real_idx = self.dof_map[sim_idx]
            self.low_cmd_buffer.motor_cmd[real_idx].mode = 0x00
            self.low_cmd_buffer.motor_cmd[real_idx].q = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].dq = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].tau = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].kp = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].kd = 0.
        self.low_cmd_buffer.crc = get_crc(self.low_cmd_buffer)
        self.low_cmd_pub.publish(self.low_cmd_buffer)
    """ Done: functions that actually publish the commands and take effect """

    def warm_up(self):
        """This warm up is useful in my experiment on Go2
        The first two iterations are very slow, but the rest is fast"""
        for _ in range(2):
            start_time = time.monotonic()

            proprio = self.get_proprio()
            get_pro_time = time.monotonic()
            proprio_history = self._get_history_proprio() 
            get_hist_pro_time = time.monotonic()

            depth_image = self._get_depth_image()
            self.depth_latent_yaw = self.depth_encode(depth_image, proprio)

            get_obs_time = time.monotonic()

            obs = self.turn_obs(proprio, self.depth_latent_yaw, proprio_history, self.n_proprio, self.n_depth_latent, self.n_hist_len)
            obs = torch.clip(obs, -self.clip_obs, self.clip_obs)

            turn_obs_time = time.monotonic()

            action = self.policy(obs)
            policy_time = time.monotonic()

            publish_time = time.monotonic()
            print("warm up: ",
                "get proprio time: {:.5f}".format(get_pro_time - start_time),
                "get hist pro time: {:.5f}".format(get_hist_pro_time - get_pro_time),
                "get_depth time: {:.5f}".format(get_obs_time - get_hist_pro_time),
                "get obs time: {:.5f}".format(get_obs_time - start_time),
                "turn_obs_time: {:.5f}".format(turn_obs_time - get_obs_time),
                "policy_time: {:.5f}".format(policy_time - turn_obs_time),
                "publish_time: {:.5f}".format(publish_time - policy_time),
                "total time: {:.5f}".format(publish_time - start_time)
            )

    def register_models(self, turn_obs, depth_encode, policy):
        """Register the model functions for observation processing and policy execution"""
        self.turn_obs = turn_obs
        self.depth_encode = depth_encode
        self.policy = policy
