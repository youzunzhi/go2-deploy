import os, sys
from typing import Optional

import rclpy
from unitree_go.msg import (
    WirelessController,
    LowState,
    LowCmd,
)
from unitree_api.msg import Request


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

from utils.hardware_cfgs import ROS_TOPICS, get_joint_limits_in_sim_order


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
    def __init__(self,
        joint_map: list,
        default_joint_pos: list,
        kp: float,
        kd: float,
        action_scale: float,
        clip_obs: float,
        clip_actions: Optional[float],
        device="cpu",
        dryrun=True, # if True, the robot will not send commands to the real robot
        enable_depth_capture=False, # if True, initialize RealSense pipeline for depth capture
        depth_resolution: Optional[tuple] = None, # (width, height) for depth image resolution
    ):
        self.device = device
        
        # Create ROS2 node instance using composition
        self.node = rclpy.create_node("go2_ros2_handler")

        # Store configuration parameters directly as attributes
        self.joint_map = joint_map
        self.default_joint_pos = torch.tensor(default_joint_pos, device=device, dtype=torch.float32)
        self.kp = kp
        self.kd = kd
        self.action_scale = action_scale
        self.clip_obs = clip_obs
        self.clip_actions = clip_actions

        self.dryrun = dryrun
        self.enable_depth_capture = enable_depth_capture
        self.depth_resolution = depth_resolution

        self.NUM_JOINTS = len(self.joint_map) # number of joints (12)

        self.joint_pos_limit_high_sim, self.joint_pos_limit_low_sim, self.torque_limit_sim = get_joint_limits_in_sim_order(self.joint_map, self.device)
        
        self.init_buffers()
        self.init_ros_communication()

        # Initialize depth handler if enabled
        self.depth_handler = None
        if self.enable_depth_capture:
            self.init_depth_handler()

    def init_ros_communication(self):
        """ after initializing the env and policy, register ros related callbacks and topics
        """

        # Low-level state subscriber
        self.low_state_sub = self.node.create_subscription(
            LowState,
            ROS_TOPICS["LOW_STATE"],
            self._low_state_callback,
            1
        )
        self.log_info("Low state subscriber started, waiting to receive low state messages.")

        # Wireless controller subscriber
        self.joy_stick_sub = self.node.create_subscription(
            WirelessController,
            ROS_TOPICS["WIRELESS_CONTROLLER"],
            self._joy_stick_callback,
            1
        )
        self.log_info("Wireless controller subscriber started, waiting to receive wireless controller messages.")

        # Low-level command publisher
        low_cmd_topic = ROS_TOPICS["LOW_CMD"] if not self.dryrun else ROS_TOPICS["LOW_CMD"] + "_dryrun_" + str(np.random.randint(0, 65535))
        self.low_cmd_pub = self.node.create_publisher(
            LowCmd,
            low_cmd_topic,
            1
        )
        self.low_cmd_buffer = LowCmd()

        # Sport mode publisher (Control the robot in built-in sport mode)
        self.sport_mode_pub = self.node.create_publisher(
            Request,
            ROS_TOPICS["SPORT_MODE"],
            1,
        )

        # Motion switcher publisher (Switch between built-in sport mode and low-level control mode)
        self.motion_switcher_pub = self.node.create_publisher(
            Request,
            ROS_TOPICS["MOTION_SWITCHER"],
            1,
        )

        # No depth image subscriber needed - we get frames directly from depth_handler

        self.log_info("ROS handlers started, waiting to recieve critical low state and wireless controller messages.")
        if not self.dryrun:
            self.log_warn(f"You are running the code in no-dryrun mode and publishing to '{low_cmd_topic}', Please keep safe.")
        else:
            self.log_warn(f"You are publishing low cmd to '{low_cmd_topic}' because of dryrun mode, Please check and be safe.")

        while rclpy.ok():
            rclpy.spin_once(self.node)
            if hasattr(self, "low_state_buffer") and hasattr(self, "joy_stick_buffer"):
                break
        self.log_info("Low state and wireless message received, the robot is ready to go.")
        
    def init_depth_handler(self):
        """Initialize the depth handler for RealSense camera capture"""
        assert self.depth_resolution is not None, "Depth resolution must be provided when depth capture is enabled"
        from rs_depth_handler import RSDepthHandler
        self.depth_handler = RSDepthHandler(output_resolution=self.depth_resolution)
        self.log_info(f"Depth handler initialized successfully with resolution {self.depth_resolution}")

    def init_buffers(self):
        self.xyyaw_command = torch.zeros(1, 3, device=self.device, dtype=torch.float32)
        self.dof_pos_ = torch.zeros(1, self.NUM_JOINTS, device=self.device, dtype=torch.float32)
        self.dof_vel_ = torch.zeros(1, self.NUM_JOINTS, device=self.device, dtype=torch.float32)
        self.actions = torch.zeros(1, self.NUM_JOINTS, device=self.device, dtype=torch.float32)    
        self.contact_filt = torch.ones((1, 4), device=self.device, dtype=torch.float32)
        self.last_contact_filt = torch.ones((1, 4), device=self.device, dtype=torch.float32)

    def reset_obs(self):
        self.xyyaw_command = torch.zeros(1, 3, device=self.device, dtype=torch.float32)
        self.actions = torch.zeros(1, self.NUM_JOINTS, device=self.device, dtype=torch.float32)    
        self.contact_filt = torch.ones((1, 4), device=self.device, dtype=torch.float32)
        self.last_contact_filt = torch.ones((1, 4), device=self.device, dtype=torch.float32)    

    """ ROS callbacks and handlers that update the buffer """

    def _low_state_callback(self, msg):
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

    # Observation retrieval methods for policy interface
    # These methods extract sensor data from ROS buffers and format them as PyTorch tensors

    def get_xyyaw_command(self):
        """Get joystick velocity command (x, y, yaw)"""
        return self.xyyaw_command
    
    def get_ang_vel_obs(self):
        """Get angular velocity from IMU gyroscope data"""
        ang_vel = torch.from_numpy(self.low_state_buffer.imu_state.gyroscope).unsqueeze(0).to(device=self.device, dtype=torch.float32)
        return ang_vel

    def get_base_rpy_obs(self):
        """Get base orientation as roll-pitch-yaw from IMU quaternion"""
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
        """Get joint positions relative to default pose"""
        return (self.dof_pos_ - self.default_joint_pos.unsqueeze(0))
    
    def get_dof_vel_obs(self):
        """Get joint velocities"""
        return self.dof_vel_
    
    def get_last_actions_obs(self):
        """Get previous action for temporal consistency"""
        return self.actions

    def get_contact_filt_obs(self):
        """Get filtered foot contact states based on force threshold"""
        for i in range(4):
            if self.low_state_buffer.foot_force[i] < 25:
                self.contact_filt[:, i] = -0.5
            else:
                self.contact_filt[:, i] = 0.5
        return self.contact_filt

    def get_depth_image(self):
        """Get depth image from RealSense camera for vision-based policies"""
        assert self.enable_depth_capture, "Depth capture is not enabled. Set enable_depth_capture=True when initializing the handler."
        assert self.depth_handler is not None, "Depth handler is not initialized."
        return self.depth_handler.get_depth_image(device=self.device)

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
        assert self.actions.shape == (1, self.NUM_JOINTS), f"Actions shape is {self.actions.shape}, expected (1, {self.NUM_JOINTS})"

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
        motor_cmd = self.low_cmd_buffer.motor_cmd
        assert hasattr(motor_cmd, '__getitem__'), "motor_cmd must be indexable"

        for sim_idx in range(self.NUM_JOINTS):
            real_idx = self.joint_map[sim_idx]
            if not self.dryrun:
                motor_cmd[real_idx].mode = 0x01 # type: ignore
            motor_cmd[real_idx].q = q_cmd_sim_order[sim_idx].item() # type: ignore
            motor_cmd[real_idx].dq = 0. # type: ignore
            motor_cmd[real_idx].tau = 0. # type: ignore
            motor_cmd[real_idx].kp = self.kp # type: ignore
            motor_cmd[real_idx].kd = self.kd # type: ignore
            
        self.low_cmd_buffer.crc = get_crc(self.low_cmd_buffer)
        self.low_cmd_pub.publish(self.low_cmd_buffer)

    def _turn_off_motors(self):
        """ Turn off the motors """
        motor_cmd = self.low_cmd_buffer.motor_cmd
        assert hasattr(motor_cmd, '__getitem__'), "motor_cmd must be indexable"
        
        for sim_idx in range(self.NUM_JOINTS):
            real_idx = self.joint_map[sim_idx]
            motor_cmd[real_idx].mode = 0x00 # type: ignore
            motor_cmd[real_idx].q = 0. # type: ignore
            motor_cmd[real_idx].dq = 0. # type: ignore
            motor_cmd[real_idx].tau = 0. # type: ignore
            motor_cmd[real_idx].kp = 0. # type: ignore
            motor_cmd[real_idx].kd = 0. # type: ignore
        self.low_cmd_buffer.crc = get_crc(self.low_cmd_buffer)
        self.low_cmd_pub.publish(self.low_cmd_buffer)
    """ Done: functions that actually publish the commands and take effect """

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
