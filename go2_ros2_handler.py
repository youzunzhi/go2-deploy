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
from nav_msgs.msg import Odometry


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

from utils.hardware_cfgs import ROS_TOPICS, get_joint_limits_in_sim_order, WirelessButtons
from utils.quaternion_utils import quat_rotate_inverse, get_euler_xyz



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
        enable_translation_capture=False, # if True, subscribe to the odometry topic for translation tracking
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
        self.enable_translation_capture = enable_translation_capture

        # Safe exit flag
        self.safe_exit_requested = False

        self.NUM_JOINTS = len(self.joint_map) # number of joints (12)
        
        self.joint_pos_limit_high_sim, self.joint_pos_limit_low_sim, self.torque_limit_sim = get_joint_limits_in_sim_order(self.joint_map, self.device)
        
        self.init_buffers()
        self.init_ros_communication()

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

        # Depth image subscriber (if depth capture is enabled)
        if self.enable_depth_capture:
            self.depth_image_sub = self.node.create_subscription(
                Float32MultiArray,
                ROS_TOPICS["DEPTH_IMAGE"],
                self._depth_image_callback,
                1
            )
            self.log_info("Depth image subscriber started, waiting to receive depth image tensors.")

        # Odometry subscriber (if translation capture is enabled)
        if self.enable_translation_capture:
            self.odometry_sub = self.node.create_subscription(
                Odometry,
                ROS_TOPICS["ODOMETRY"],
                self._odometry_callback,
                1
            )
            self.log_info(f"Odometry subscriber started, waiting to receive odometry messages from {ROS_TOPICS['ODOMETRY']}.")

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
        
    def init_buffers(self):
        self.xyyaw_command = torch.zeros(1, 3, device=self.device, dtype=torch.float32)
        self.dof_pos_ = torch.zeros(1, self.NUM_JOINTS, device=self.device, dtype=torch.float32)
        self.dof_vel_ = torch.zeros(1, self.NUM_JOINTS, device=self.device, dtype=torch.float32)
        self.actions = torch.zeros(1, self.NUM_JOINTS, device=self.device, dtype=torch.float32)
        self.contact_filt = torch.ones((1, 4), device=self.device, dtype=torch.float32)
        self.last_contact_filt = torch.ones((1, 4), device=self.device, dtype=torch.float32)
        if self.enable_depth_capture:
            self.depth_tensor = torch.zeros(1, self.depth_resolution[0], self.depth_resolution[1], device=self.device, dtype=torch.float32)
        if self.enable_translation_capture:
            # Translation tracking buffers - positions are (x, y, z) in meters
            self.start_pos = torch.zeros(1, 3, device=self.device, dtype=torch.float32)
            self.cur_pos = torch.zeros(1, 3, device=self.device, dtype=torch.float32)
            # Store initial robot orientation for robot-frame translation calculation
            self.start_quat = torch.zeros(1, 4, device=self.device, dtype=torch.float32)  # xyzw format
            self.start_pos_captured = False

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
        
        # Check for select button press for safe exit
        if msg.keys & WirelessButtons.select:
            self.log_warn("SELECT button pressed - initiating safe exit!")
            self.safe_exit_requested = True
        
    def _depth_image_callback(self, msg):
        """Callback for receiving depth image tensors from depth publisher node"""
        try:
            # Check for error signal (empty data)
            if len(msg.data) == 0:
                self.log_error("Received empty depth tensor - depth capture has failed!")
                self.depth_tensor = None
                return
            
            # Reconstruct tensor from Float32MultiArray message using known shape
            # Shape is (1, height, width) where height, width come from depth_resolution
            height, width = self.depth_resolution[1], self.depth_resolution[0]  # depth_resolution is (width, height)
            expected_shape = (1, height, width)
            
            # Convert flat list back to tensor with known shape
            self.depth_tensor = torch.tensor(msg.data, dtype=torch.float32).reshape(expected_shape).to(self.device)
            
        except Exception as e:
            self.log_error(f"Error processing depth image message: {e}")
            self.depth_tensor = None

    def _odometry_callback(self, msg):
        """Callback for receiving odometry data from odometry topic"""
        try:
            # Extract position from odometry message
            position = msg.pose.pose.position
            current_pos = torch.tensor([[position.x, position.y, position.z]],
                                     device=self.device, dtype=torch.float32)

            # Update current pose
            self.cur_pos = current_pos

            # Capture start position and orientation on first message
            if not self.start_pos_captured:
                self.start_pos = current_pos.clone()

                # Extract and store initial robot orientation from odometry
                orientation = msg.pose.pose.orientation
                self.start_quat = torch.tensor([[orientation.x, orientation.y, orientation.z, orientation.w]],
                                             device=self.device, dtype=torch.float32)

                self.start_pos_captured = True
                self.log_info(f"Translation capture: recorded start position at [{position.x:.3f}, {position.y:.3f}, {position.z:.3f}]")
                self.log_info(f"Translation capture: recorded start orientation quat [x:{orientation.x:.3f}, y:{orientation.y:.3f}, z:{orientation.z:.3f}, w:{orientation.w:.3f}]")

        except Exception as e:
            self.log_error(f"Error processing odometry message: {e}")

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
        quat_xyzw = self.get_base_quat_obs()
        roll, pitch, yaw = get_euler_xyz(quat_xyzw)
        base_rpy = torch.tensor([[roll, pitch, yaw]], device=self.device, dtype=torch.float32)
        return base_rpy

    def get_base_quat_obs(self):
        """Get base orientation quaternion in xyzw order (normalized)"""
        q_wxyz = self.low_state_buffer.imu_state.quaternion
        quat_xyzw = torch.tensor([[q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]],
                                 device=self.device, dtype=torch.float32)
        return quat_xyzw

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
        """Get latest depth image tensor (non-blocking)
        
        Returns:
            torch.Tensor: Latest depth image tensor
        """
        assert self.enable_depth_capture, "Depth capture is not enabled."
        assert self.depth_tensor is not None, "Depth capture has failed - cannot continue vision-based policy"
        
        return self.depth_tensor.clone()  # Return a copy to avoid race conditions
    
    def get_translation(self):
        """Get translation (position difference) from start position to current position in robot frame

        Returns:
            torch.Tensor: Translation vector (1, 3) in meters [x, y, z] in robot frame
                         where x=forward, y=left, z=up relative to robot's initial orientation
        """
        assert self.enable_translation_capture, "Translation capture is not enabled."
        assert self.start_pos_captured, "Start position has not been captured yet - no odometry data received"

        # Get world-frame translation
        world_translation = self.cur_pos - self.start_pos

        # Transform world-frame translation to robot frame using initial robot orientation
        # Use inverse rotation to transform from world frame to robot frame
        robot_translation = quat_rotate_inverse(self.start_quat, world_translation)

        return robot_translation

    def get_translation_world_frame(self):
        """Get translation (position difference) from start position to current position in world frame

        Returns:
            torch.Tensor: Translation vector (1, 3) in meters [x, y, z] in world frame
        """
        assert self.enable_translation_capture, "Translation capture is not enabled."
        assert self.start_pos_captured, "Start position has not been captured yet - no odometry data received"

        # Return current position relative to start position in world frame
        world_translation = self.cur_pos - self.start_pos
        return world_translation

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
    
    def safe_exit(self):
        """Emergency safe exit - turn off motors and signal program to exit"""
        self.log_warn("Executing safe exit - turning off motors")
        self._turn_off_motors()
        self.log_info("Motors turned off safely")
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
