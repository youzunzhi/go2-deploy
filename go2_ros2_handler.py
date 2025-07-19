import os, sys
from typing import Optional

import rclpy
from unitree_go.msg import (
    WirelessController,
    LowState,
    LowCmd,
)
from unitree_api.msg import Request

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

import numpy as np
import torch
import time

from utils.hardware import JOINT_POS_LIMIT_HIGH, JOINT_POS_LIMIT_LOW, TORQUE_LIMIT


@torch.jit.script  # type: ignore
def copysign(a: float, b: torch.Tensor) -> torch.Tensor:
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])  # type: ignore
    return torch.abs(a) * torch.sign(b)  # type: ignore

@torch.jit.script  # type: ignore
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

    

class Go2ROS2Handler:
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
        joint_map: list,
        default_joint_pos: list,
        kp: float,
        kd: float,
        action_scale: float,
        clip_obs: float,
        clip_actions: Optional[float],
        device="cpu",
        dof_pos_protect_ratio=1.1, # if the dof_pos is out of the range of this ratio, the process will shutdown.
        dryrun=True, # if True, the robot will not send commands to the real robot
        mode="locomotion",
        policy_source="EPO", # Policy source: "EPO" or "legged-loco"
    ):
        self.device = device
        
        # Create ROS2 node instance using composition
        self.node = rclpy.create_node("unitree_ros2_real")

        # Store configuration parameters directly as attributes
        self.joint_map = joint_map
        self.default_joint_pos = torch.tensor(default_joint_pos, device=device, dtype=torch.float32)
        self.kp = kp
        self.kd = kd
        self.action_scale = action_scale
        self.clip_obs = clip_obs
        self.clip_actions = clip_actions

        self.low_state_topic = "/lowstate"
        self.low_cmd_topic = "/lowcmd" if not dryrun else "/lowcmd_dryrun_" + str(np.random.randint(0, 65535))
        self.joy_stick_topic = "/wirelesscontroller"
        self.depth_data_topic = "/forward_depth_image"

        self.dryrun = dryrun

        self.NUM_JOINTS = len(self.joint_map) # number of joints (12)

        self.xyyaw_command = torch.tensor([[0, 0, 0]], device=self.device, dtype=torch.float32)
        self.contact_filt = torch.ones((1, 4), device=self.device, dtype=torch.float32)
        self.last_contact_filt = torch.ones((1, 4), device=self.device, dtype=torch.float32)
        self.dof_pos_ = torch.empty(1, self.NUM_JOINTS, device=self.device, dtype=torch.float32)
        self.dof_vel_ = torch.empty(1, self.NUM_JOINTS, device=self.device, dtype=torch.float32)
        self.actions = torch.zeros(self.NUM_JOINTS, device=self.device, dtype=torch.float32)    

        ###################### hardware related #####################
        # Use policy-specific joint and torque limits
        joint_pos_limit_high_real = list(JOINT_POS_LIMIT_HIGH.values())
        joint_pos_limit_low_real = list(JOINT_POS_LIMIT_LOW.values())
        torque_limit_real = list(TORQUE_LIMIT.values())
        joint_pos_limit_high_sim = self.map_list_in_real_order_to_sim_order(joint_pos_limit_high_real)
        joint_pos_limit_low_sim = self.map_list_in_real_order_to_sim_order(joint_pos_limit_low_real)
        torque_limit_sim = self.map_list_in_real_order_to_sim_order(torque_limit_real)
        self.joint_pos_limit_high_sim = torch.tensor(joint_pos_limit_high_sim, device=self.device, dtype=torch.float32)
        self.joint_pos_limit_low_sim = torch.tensor(joint_pos_limit_low_sim, device=self.device, dtype=torch.float32)
        self.torque_limit_sim = torch.tensor(torque_limit_sim, device=self.device, dtype=torch.float32)
        
        self.visual_update_interval = 5

    def map_list_in_real_order_to_sim_order(self, list_in_real_order: list) -> list:
        """
        Map a list of configs in real joint order to sim joint order
        """
        return [list_in_real_order[self.joint_map[sim_idx]] for sim_idx in range(len(list_in_real_order))]
        
    def start_ros_handlers(self):
        """ after initializing the env and policy, register ros related callbacks and topics
        """
        # Low-level command publisher
        self.low_cmd_pub = self.node.create_publisher(
            LowCmd,
            self.low_cmd_topic,
            1
        )
        self.low_cmd_buffer = LowCmd()

        # Low-level state subscriber
        self.low_state_sub = self.node.create_subscription(
            LowState,
            self.low_state_topic,
            self._low_state_callback,
            1
        )
        self.log_info("Low state subscriber started, waiting to receive low state messages.")

        # Wireless controller subscriber
        self.joy_stick_sub = self.node.create_subscription(
            WirelessController,
            self.joy_stick_topic,
            self._joy_stick_callback,
            1
        )
        self.log_info("Wireless controller subscriber started, waiting to receive wireless controller messages.")

        # Sport mode publisher (Control the robot in built-in sport mode)
        self.sport_mode_pub = self.node.create_publisher(
            Request,
            '/api/sport/request',
            1,
        )

        # Motion switcher publisher (Switch between built-in sport mode and low-level control mode)
        self.motion_switcher_pub = self.node.create_publisher(
            Request,
            '/api/motion_switcher/request',
            1,
        )

        # Depth image subscriber (For EPO policy)
        self.depth_input_sub = self.node.create_subscription(
            Float32MultiArray,
            self.depth_data_topic,
            self._depth_data_callback,
            1
        )

        self.log_info("ROS handlers started, waiting to recieve critical low state and wireless controller messages.")
        if not self.dryrun:
            self.log_warn(f"You are running the code in no-dryrun mode and publishing to '{self.low_cmd_topic}', Please keep safe.")
        else:
            self.log_warn(f"You are publishing low cmd to '{self.low_cmd_topic}' because of dryrun mode, Please check and be safe.")
        while rclpy.ok():
            rclpy.spin_once(self.node)
            if hasattr(self, "low_state_buffer") and hasattr(self, "joy_stick_buffer"):
                break
        self.log_info("Low state and wireless message received, the robot is ready to go.")

    def reset_obs(self):
        self.actions = torch.zeros(self.NUM_JOINTS, device=self.device, dtype=torch.float32)    
        self.xyyaw_command = torch.tensor([[0, 0, 0]], device=self.device, dtype=torch.float32)
        self.contact_filt = torch.ones((1, 4), device=self.device, dtype=torch.float32)
        self.last_contact_filt = torch.ones((1, 4), device=self.device, dtype=torch.float32)    

    """ ROS callbacks and handlers that update the buffer """

    def _low_state_callback(self, msg):
        # self.get_logger().warn("Low state message received.")
        """ store and handle proprioception data """
        self.low_state_buffer = msg # keep the latest low state

        ################### refresh dof_pos and dof_vel ######################
        for sim_idx in range(self.NUM_JOINTS):
            real_idx = self.joint_map[sim_idx]
            self.dof_pos_[0, sim_idx] = self.low_state_buffer.motor_state[real_idx].q
        for sim_idx in range(self.NUM_JOINTS):
            real_idx = self.joint_map[sim_idx]
            self.dof_vel_[0, sim_idx] = self.low_state_buffer.motor_state[real_idx].dq

    def _joy_stick_callback(self, msg):
        # Configurable parameters for the joy stick
        lin_vel_deadband = 0.1
        ang_vel_deadband = 0.1
        cmd_px_range = [0.4, 1.0] 
        cmd_nx_range = [0.4, 0.8]
        cmd_py_range = [0.4, 0.8]
        cmd_ny_range = [0.4, 0.8]
        cmd_pyaw_range = [0.4, 1.6] 
        cmd_nyaw_range = [0.4, 1.6] 
        
        # Update the buffer
        self.joy_stick_buffer = msg

        # Process the message
        # left-y for forward/backward
        ly = msg.ly
        if ly > lin_vel_deadband:
            vx = (ly - lin_vel_deadband) / (1 - lin_vel_deadband) # (0, 1)
            vx = vx * (cmd_px_range[1] - cmd_px_range[0]) + cmd_px_range[0]
        elif ly < -lin_vel_deadband:
            vx = (ly + lin_vel_deadband) / (1 - lin_vel_deadband) # (-1, 0)
            vx = vx * (cmd_nx_range[1] - cmd_nx_range[0]) - cmd_nx_range[0]
        else:
            vx = 0
        # left-x for turning left/right
        lx = -msg.lx
        if lx > ang_vel_deadband:
            yaw = (lx - ang_vel_deadband) / (1 - ang_vel_deadband)
            yaw = yaw * (cmd_pyaw_range[1] - cmd_pyaw_range[0]) + cmd_pyaw_range[0]
        elif lx < -ang_vel_deadband:
            yaw = (lx + ang_vel_deadband) / (1 - ang_vel_deadband)
            yaw = yaw * (cmd_nyaw_range[1] - cmd_nyaw_range[0]) - cmd_nyaw_range[0]
        else:
            yaw = 0
        # right-x for side moving left/right
        rx = -msg.rx
        if rx > lin_vel_deadband:
            vy = (rx - lin_vel_deadband) / (1 - lin_vel_deadband)
            vy = vy * (cmd_py_range[1] - cmd_py_range[0]) + cmd_py_range[0]
        elif rx < -lin_vel_deadband:
            vy = (rx + lin_vel_deadband) / (1 - lin_vel_deadband)
            vy = vy * (cmd_ny_range[1] - cmd_ny_range[0]) - cmd_ny_range[0]
        else:
            vy = 0

        # Update the buffer
        self.xyyaw_command = torch.tensor([[vx, vy, yaw]], device=self.device, dtype=torch.float32)

    def _depth_data_callback(self, msg):
        self.depth_data = torch.tensor(msg.data, dtype=torch.float32).reshape(1, 58, 87).to(self.device)

    def _sport_mode_command(self, api_id):
        msg = Request()

        msg.header.identity.id = 0
        msg.header.identity.api_id = api_id
        msg.header.lease.id = 0
        msg.header.policy.priority = 0
        msg.header.policy.noreply = False

        msg.parameter = ''
        msg.binary = []

        self.sport_mode_pub.publish(msg)
    
    def _sport_mode_switch(self, mode):
        msg = Request()

        # Fill the header
        msg.header.identity.id = 0
        msg.header.lease.id = 0
        msg.header.policy.priority = 0
        msg.header.policy.noreply = False

        if mode == 0:
            # Release mode (switch to low-level control mode) - use api_id 1003
            msg.header.identity.api_id = 1003
            msg.parameter = '{}'
        elif mode == 1:
            # Select sport mode - use api_id 1002
            msg.header.identity.api_id = 1002
            msg.parameter = '{"name": "mcf"}'
        
        msg.binary = []

        # Publish to motion switcher instead of robot state
        self.motion_switcher_pub.publish(msg)

    """ Done: ROS callbacks and handlers that update the buffer """

    """ refresh observation buffer and corresponding sub-functions """

    def get_xyyaw_command(self):
        return self.xyyaw_command
    
    def get_ang_vel_obs(self):
        ang_vel = torch.from_numpy(self.low_state_buffer.imu_state.gyroscope).unsqueeze(0).to(device=self.device, dtype=torch.float32)
        return ang_vel

    def get_base_rpy_obs(self):
        quat_xyzw = torch.tensor([
            self.low_state_buffer.imu_state.quaternion[1],
            self.low_state_buffer.imu_state.quaternion[2],
            self.low_state_buffer.imu_state.quaternion[3],
            self.low_state_buffer.imu_state.quaternion[0],
            ], device=self.device, dtype=torch.float32).unsqueeze(0)
        roll, pitch, yaw = get_euler_xyz(quat_xyzw)
        base_rpy = torch.tensor([[roll, pitch, yaw]], device=self.device, dtype=torch.float32)
        return base_rpy

    def get_dof_pos_obs(self):
        return (self.dof_pos_ - self.default_joint_pos.unsqueeze(0))
    
    def get_dof_vel_obs(self):
        return self.dof_vel_
    
    def get_last_actions_obs(self):
        return self.actions

    def get_contact_filt_obs(self):
        for i in range(4):
            if self.low_state_buffer.foot_force[i] < 25:
                self.contact_filt[:, i] = -0.5
            else:
                self.contact_filt[:, i] = 0.5
        return self.contact_filt

    def get_depth_image(self):
        return self.depth_data

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
                                   self.joint_pos_limit_low_sim.unsqueeze(0), 
                                   self.joint_pos_limit_high_sim.unsqueeze(0))
        
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
        torque_limit = self.torque_limit_sim.unsqueeze(0)  # (1, NUM_DOF)
        
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
            actions = torch.tensor(actions, device=self.device).unsqueeze(0)
        
        self.actions = actions

        if self.clip_actions is not None:
            hard_clip = self.clip_actions / self.action_scale
            actions = torch.clip(actions, -hard_clip, hard_clip)
        scaled_actions = actions * self.action_scale
        robot_coordinates_action = scaled_actions + self.default_joint_pos.unsqueeze(0)
        
        # Apply safety clipping using joint limits
        robot_coordinates_action = self.clip_actions_by_joint_limits(robot_coordinates_action)
        
        # Apply safety clipping using torque limits
        current_joint_pos = self.dof_pos_
        current_joint_vel = self.dof_vel_
        robot_coordinates_action = self.clip_actions_by_torque_limits(robot_coordinates_action, current_joint_pos, current_joint_vel)
        
        self._publish_legs_cmd(robot_coordinates_action[0])

    """ functions that actually publish the commands and take effect """
    def _publish_legs_cmd(self, q_cmd_sim_order):
        """ Publish the joint commands to the robot legs in simulation order.
        q_cmd_sim_order: shape (NUM_DOF,), in simulation order.
        """
        #################### check ##############################
        for sim_idx in range(self.NUM_JOINTS):
            real_idx = self.joint_map[sim_idx]
            if not self.dryrun:
                self.low_cmd_buffer.motor_cmd[real_idx].mode = 0x01
            self.low_cmd_buffer.motor_cmd[real_idx].q = q_cmd_sim_order[sim_idx].item()
            self.low_cmd_buffer.motor_cmd[real_idx].dq = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].tau = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].kp = self.kp
            self.low_cmd_buffer.motor_cmd[real_idx].kd = self.kd
        
        self.low_cmd_buffer.crc = get_crc(self.low_cmd_buffer)
        self.low_cmd_pub.publish(self.low_cmd_buffer)

    def _turn_off_motors(self):
        """ Turn off the motors """
        for sim_idx in range(self.NUM_JOINTS):
            real_idx = self.joint_map[sim_idx]
            self.low_cmd_buffer.motor_cmd[real_idx].mode = 0x00
            self.low_cmd_buffer.motor_cmd[real_idx].q = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].dq = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].tau = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].kp = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].kd = 0.
        self.low_cmd_buffer.crc = get_crc(self.low_cmd_buffer)
        self.low_cmd_pub.publish(self.low_cmd_buffer)
    """ Done: functions that actually publish the commands and take effect """

    def register_models(self, turn_obs, depth_encode, policy):
        """Register the model functions for observation processing and policy execution"""
        self.turn_obs = turn_obs
        self.depth_encode = depth_encode
        self.policy = policy
    
    def log_info(self, message, **kwargs):
        """Convenient logging method for info messages"""
        self.node.get_logger().info(message, **kwargs)
    
    def log_warn(self, message, **kwargs):
        """Convenient logging method for warning messages"""
        self.node.get_logger().warn(message, **kwargs)
    
    def log_error(self, message, **kwargs):
        """Convenient logging method for error messages"""
        self.node.get_logger().error(message, **kwargs)
    
    def shutdown(self):
        """Shutdown the ROS2 node properly"""
        if hasattr(self, 'node'):
            self.node.destroy_node()
