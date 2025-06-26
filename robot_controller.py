# robot_controller.py
# 机器人控制基类

import rclpy
from rclpy.node import Node
import torch
import numpy as np
import time
from typing import Optional, Dict, Any, List, Tuple

from constants import RobotConfig, ObservationConfig, ControlConfig, RunMode
from utils import get_euler_xyz, get_crc, PerformanceTimer, StateMachine, SafetyChecker
from config import RobotConfiguration, DeploymentConfig
from inference_engine import InferenceEngine


class WirelessButtons:
    """无线手柄按钮定义"""
    R1 = 0b00000001  # 1
    L1 = 0b00000010  # 2
    START = 0b00000100  # 4
    SELECT = 0b00001000  # 8
    R2 = 0b00010000  # 16
    L2 = 0b00100000  # 32
    F1 = 0b01000000  # 64
    F2 = 0b10000000  # 128
    A = 0b100000000  # 256
    B = 0b1000000000  # 512
    X = 0b10000000000  # 1024
    Y = 0b100000000000  # 2048
    UP = 0b1000000000000  # 4096
    RIGHT = 0b10000000000000  # 8192
    DOWN = 0b100000000000000  # 16384
    LEFT = 0b1000000000000000  # 32768


class RobotController(Node):
    """机器人控制基类"""
    
    def __init__(self, 
                 robot_name: str = "go2",
                 config: Optional[RobotConfiguration] = None,
                 deploy_config: Optional[DeploymentConfig] = None,
                 **kwargs):
        """
        初始化机器人控制器
        
        Args:
            robot_name: 机器人名称
            config: 机器人配置
            deploy_config: 部署配置
        """
        super().__init__(f"{robot_name}_controller")
        
        # 配置管理
        self.config = config or RobotConfiguration()
        self.deploy_config = deploy_config or DeploymentConfig()
        
        # 机器人参数
        self.robot_name = robot_name
        self.num_dof = RobotConfig.num_dof
        self.num_actions = RobotConfig.num_actions
        self.dof_map = RobotConfig.dof_map
        self.dof_names = RobotConfig.dof_names
        self.dof_signs = RobotConfig.dof_signs
        self.turn_on_motor_mode = RobotConfig.turn_on_motor_mode
        
        # 设备配置
        self.device = self.deploy_config.device
        self.dryrun = self.deploy_config.dryrun
        
        # 观察空间配置
        self.n_proprio = ObservationConfig.n_proprio
        self.n_depth_latent = ObservationConfig.n_depth_latent
        self.n_hist_len = ObservationConfig.n_hist_len
        
        # 状态管理
        self.state_machine = StateMachine("sport_mode")
        self.performance_timer = PerformanceTimer()
        
        # 安全检查器
        self.safety_checker = SafetyChecker(
            RobotConfig.joint_limits_low,
            RobotConfig.joint_limits_high,
            self.deploy_config.safety_ratio
        )
        
        # 推理引擎
        self.inference_engine = None
        if hasattr(self.deploy_config, 'model_dir') and self.deploy_config.model_dir:
            self.inference_engine = InferenceEngine(
                self.deploy_config.model_dir,
                self.device,
                self.deploy_config.warm_up_iterations
            )
        
        # ROS接口
        self.ros_interface = None
        
        # 控制参数
        self.lin_vel_deadband = ControlConfig.lin_vel_deadband
        self.ang_vel_deadband = ControlConfig.ang_vel_deadband
        self.cmd_px_range = ControlConfig.cmd_px_range
        self.cmd_nx_range = ControlConfig.cmd_nx_range
        self.cmd_py_range = ControlConfig.cmd_py_range
        self.cmd_ny_range = ControlConfig.cmd_ny_range
        self.cmd_pyaw_range = ControlConfig.cmd_pyaw_range
        self.cmd_nyaw_range = ControlConfig.cmd_nyaw_range
        
        # 数据缓冲区
        self._init_buffers()
        
        # 控制参数
        self._init_control_params()
        
        # 站立配置
        self._init_stand_config()
        
        # 安全状态
        self.safety_violations = 0
        self.max_safety_violations = 10
        
        # 性能监控
        self.control_cycle_count = 0
        self.last_control_time = time.monotonic()
        
        # 推理相关
        self.global_counter = 0
        self.visual_update_interval = ControlConfig.visual_update_interval
        
        self.get_logger().info(f"机器人控制器初始化完成: {robot_name}")
        self.get_logger().info(f"设备: {self.device}, 干运行模式: {self.dryrun}")
    
    def _init_buffers(self):
        """初始化数据缓冲区"""
        # 观察缓冲区
        self.proprio_history_buf = torch.zeros(
            1, self.n_hist_len, self.n_proprio, 
            device=self.device, dtype=torch.float
        )
        self.episode_length_buf = torch.zeros(
            1, device=self.device, dtype=torch.float
        )
        self.forward_depth_latent_yaw_buffer = torch.zeros(
            1, self.n_depth_latent + 2, 
            device=self.device, dtype=torch.float
        )
        
        # 控制缓冲区
        self.xyyaw_command = torch.tensor(
            [[0, 0, 0]], device=self.device, dtype=torch.float32
        )
        self.contact_filt = torch.ones(
            (1, 4), device=self.device, dtype=torch.float32
        )
        self.last_contact_filt = torch.ones(
            (1, 4), device=self.device, dtype=torch.float32
        )
        
        # 关节状态缓冲区
        self.dof_pos_ = torch.empty(
            1, self.num_dof, device=self.device, dtype=torch.float32
        )
        self.dof_vel_ = torch.empty(
            1, self.num_dof, device=self.device, dtype=torch.float32
        )
        self.actions = torch.zeros(
            self.num_actions, device=self.device, dtype=torch.float32
        )
        
        # 深度图像缓冲区
        self.depth_data = None
        self.last_depth_image = None
    
    def _init_control_params(self):
        """初始化控制参数"""
        # 从配置获取增益
        self.p_gains = torch.tensor(
            self.config.get_stiffness_gains(), 
            device=self.device, dtype=torch.float32
        )
        self.d_gains = torch.tensor(
            self.config.get_damping_gains(), 
            device=self.device, dtype=torch.float32
        )
        
        # 默认关节位置
        default_angles = self.config.get_default_joint_angles()
        self.default_dof_pos = torch.zeros(
            self.num_dof, device=self.device, dtype=torch.float32
        )
        for i, name in enumerate(self.dof_names):
            if name in default_angles:
                self.default_dof_pos[i] = default_angles[name]
        
        # 动作限制
        self.action_scale = self.config.get("control.action_scale", 1.0)
        self.clip_actions = self.config.get("normalization.clip_actions", 1.0)
        
        # 观察缩放
        self.obs_scales = self.config.get("normalization.obs_scales", {})
    
    def _init_stand_config(self):
        """初始化站立配置"""
        self.start_pos = [0.0] * self.num_dof
        self._target_pos_1 = [0.0, 1.36, -2.65, 0.0, 1.36, -2.65,
                             -0.2, 1.36, -2.65, 0.2, 1.36, -2.65]
        self._target_pos_2 = [0.0, 0.67, -1.3, 0.0, 0.67, -1.3,
                             0.0, 0.67, -1.3, 0.0, 0.67, -1.3]
        self.stand_action = [0.0] * self.num_dof
        
        self.duration_1 = 10
        self.duration_2 = 100
        self.percent_1 = 0
        self.percent_2 = 0
        
        self.first_run_target_1 = True
        self.first_run = True
    
    def set_ros_interface(self, ros_interface):
        """设置ROS接口"""
        self.ros_interface = ros_interface
        self.get_logger().info("ROS接口已设置")
    
    def set_inference_engine(self, inference_engine):
        """设置推理引擎"""
        self.inference_engine = inference_engine
        self.get_logger().info("推理引擎已设置")
    
    def switch_to_sport_mode(self):
        """切换到运动模式"""
        if self.state_machine.transition_to("sport_mode"):
            self.get_logger().info("切换到运动模式")
            
            # 通过ROS接口发布运动模式切换命令
            if self.ros_interface:
                self.ros_interface.publish_motion_switcher(1)  # 选择MCF模式
            
            return True
        return False
    
    def switch_to_normal_mode(self):
        """切换到普通模式"""
        if self.state_machine.transition_to("normal_mode"):
            self.get_logger().info("切换到普通模式")
            
            # 通过ROS接口发布运动模式切换命令
            if self.ros_interface:
                self.ros_interface.publish_motion_switcher(0)  # 释放模式
            
            return True
        return False
    
    def publish_sport_mode_command(self, mode_id: int):
        """发布运动模式命令"""
        if self.ros_interface:
            self.ros_interface.publish_sport_mode(mode_id)
            self.get_logger().info(f"发布运动模式命令: {mode_id}")
    
    def initialize_inference_engine(self) -> bool:
        """
        初始化推理引擎
        
        Returns:
            bool: 是否成功初始化
        """
        if self.inference_engine is None:
            self.get_logger().warn("推理引擎未配置")
            return False
        
        try:
            success = self.inference_engine.initialize()
            if success:
                self.get_logger().info("推理引擎初始化成功")
            else:
                self.get_logger().error("推理引擎初始化失败")
            return success
        except Exception as e:
            self.get_logger().error(f"推理引擎初始化异常: {e}")
            return False
    
    def check_safety(self, joint_positions: torch.Tensor) -> bool:
        """
        安全检查
        
        Args:
            joint_positions: 关节位置
            
        Returns:
            bool: 是否安全
        """
        if not self.deploy_config.enable_safety_check:
            return True
        
        is_safe = self.safety_checker.check_joint_positions(joint_positions)
        
        if not is_safe:
            self.safety_violations += 1
            violation_info = self.safety_checker.get_violation_info(joint_positions)
            
            self.get_logger().warn(
                f"安全检查失败! 违规次数: {self.safety_violations}/{self.max_safety_violations}"
            )
            self.get_logger().warn(f"违规关节: {violation_info['unsafe_joints']}")
            
            if self.safety_violations >= self.max_safety_violations:
                self.get_logger().error("达到最大安全违规次数，启动紧急停止!")
                self.emergency_stop()
                return False
        
        return is_safe
    
    def emergency_stop(self):
        """紧急停止"""
        self.get_logger().error("执行紧急停止!")
        
        # 切换到安全状态
        self.state_machine.transition_to("emergency")
        
        # 关闭电机
        self._turn_off_motors()
        
        # 重置安全计数器
        self.safety_violations = 0
    
    def _turn_off_motors(self):
        """关闭电机"""
        self.get_logger().info("关闭所有电机")
        # TODO: 实现具体的电机关闭逻辑
    
    def get_proprio(self) -> torch.Tensor:
        """
        获取本体感受信息
        
        Returns:
            torch.Tensor: 本体感受观察向量
        """
        self.performance_timer.start()
        
        # 获取角速度
        ang_vel = self._get_ang_vel_obs()
        self.performance_timer.checkpoint("ang_vel")
        
        # 获取IMU信息
        imu = self._get_imu_obs()
        self.performance_timer.checkpoint("imu")
        
        # 获取偏航角信息
        yaw_info = self._get_delta_yaw_obs()
        self.performance_timer.checkpoint("yaw")
        
        # 获取命令
        commands = self._get_commands_obs()
        self.performance_timer.checkpoint("commands")
        
        # 获取模式标识
        mode_flag = self._get_mode_flag()
        self.performance_timer.checkpoint("mode")
        
        # 获取关节位置
        dof_pos = self._get_dof_pos_obs()
        self.performance_timer.checkpoint("dof_pos")
        
        # 获取关节速度
        dof_vel = self._get_dof_vel_obs()
        self.performance_timer.checkpoint("dof_vel")
        
        # 获取上一动作
        last_actions = self._get_last_actions_obs()
        self.performance_timer.checkpoint("last_actions")
        
        # 获取接触状态
        contact = self._get_contact_filt_obs()
        self.performance_timer.checkpoint("contact")
        
        # 组合观察向量
        proprio = torch.cat([
            ang_vel, imu, yaw_info, commands, mode_flag,
            dof_pos, dof_vel, last_actions, contact
        ], dim=-1)
        
        # 更新历史缓冲区
        self._update_proprio_history(proprio)
        
        self.performance_timer.checkpoint("total_proprio")
        
        return proprio
    
    def _get_ang_vel_obs(self) -> torch.Tensor:
        """获取角速度观察"""
        # TODO: 从实际的IMU数据获取
        ang_vel = torch.zeros(1, 3, device=self.device, dtype=torch.float32)
        return ang_vel * self.obs_scales.get("ang_vel", 0.25)
    
    def _get_imu_obs(self) -> torch.Tensor:
        """获取IMU观察"""
        # TODO: 从实际的IMU数据获取
        imu = torch.zeros(1, 2, device=self.device, dtype=torch.float32)
        return imu
    
    def _get_delta_yaw_obs(self) -> torch.Tensor:
        """获取偏航角信息"""
        # TODO: 实现偏航角计算
        yaw_info = torch.zeros(1, 3, device=self.device, dtype=torch.float32)
        return yaw_info
    
    def _get_commands_obs(self) -> torch.Tensor:
        """获取命令观察"""
        vx, _, _ = self.xyyaw_command[0, :]
        commands = torch.tensor([[0, 0, vx]], device=self.device, dtype=torch.float32)
        return commands
    
    def _get_mode_flag(self) -> torch.Tensor:
        """获取模式标识"""
        current_state = self.state_machine.get_current_state()
        if current_state == "locomotion":
            return torch.tensor([[1, 0]], device=self.device, dtype=torch.float32)
        else:
            return torch.tensor([[0, 1]], device=self.device, dtype=torch.float32)
    
    def _get_dof_pos_obs(self) -> torch.Tensor:
        """获取关节位置观察"""
        return (self.dof_pos_ - self.default_dof_pos.unsqueeze(0)) * self.obs_scales.get("dof_pos", 1.0)
    
    def _get_dof_vel_obs(self) -> torch.Tensor:
        """获取关节速度观察"""
        return self.dof_vel_ * self.obs_scales.get("dof_vel", 0.05)
    
    def _get_last_actions_obs(self) -> torch.Tensor:
        """获取上一动作观察"""
        return self.actions.view(1, -1)
    
    def _get_contact_filt_obs(self) -> torch.Tensor:
        """获取接触状态观察"""
        # TODO: 从实际的足部力传感器获取
        return self.contact_filt
    
    def _update_proprio_history(self, proprio: torch.Tensor):
        """更新本体感受历史"""
        self.proprio_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([proprio] * self.n_hist_len, dim=1),
            torch.cat([
                self.proprio_history_buf[:, 1:],
                proprio.unsqueeze(1)
            ], dim=1)
        )
        self.episode_length_buf += 1
    
    def get_history_proprio(self) -> torch.Tensor:
        """获取历史本体感受"""
        return self.proprio_history_buf
    
    def get_depth_image(self) -> Optional[torch.Tensor]:
        """获取深度图像"""
        return self.depth_data
    
    def execute_locomotion_policy(self) -> torch.Tensor:
        """
        执行运动控制策略
        
        Returns:
            torch.Tensor: 动作
        """
        if self.inference_engine is None:
            self.get_logger().error("推理引擎未初始化")
            return torch.zeros(self.num_actions, device=self.device, dtype=torch.float32)
        
        start_time = time.monotonic()
        
        # 获取本体感受
        proprio = self.get_proprio()
        get_pro_time = time.monotonic()
        
        # 获取历史本体感受
        proprio_history = self.get_history_proprio()
        get_hist_pro_time = time.monotonic()
        
        # 处理深度图像
        depth_image = None
        if self.global_counter % self.visual_update_interval == 0:
            depth_image = self.get_depth_image()
            if self.global_counter == 0:
                self.last_depth_image = depth_image
        
        # 执行推理
        action = self.inference_engine.inference_step(
            proprio, depth_image, self.global_counter
        )
        
        policy_time = time.monotonic()
        
        # 记录性能
        self.get_logger().debug(
            f"推理性能 - 本体感受: {(get_pro_time - start_time)*1000:.2f}ms, "
            f"历史: {(get_hist_pro_time - get_pro_time)*1000:.2f}ms, "
            f"推理: {(policy_time - get_hist_pro_time)*1000:.2f}ms, "
            f"总计: {(policy_time - start_time)*1000:.2f}ms"
        )
        
        self.global_counter += 1
        
        return action
    
    def send_action(self, actions: torch.Tensor):
        """
        发送动作到机器人
        
        Args:
            actions: 动作张量
        """
        if isinstance(actions, list):
            actions = torch.tensor(actions, device=self.device).unsqueeze(0)
        
        self.actions = actions
        
        # 动作裁剪和缩放
        hard_clip = self.clip_actions / self.action_scale
        clipped_scaled_action = torch.clip(actions, -hard_clip, hard_clip) * self.action_scale
        
        # 转换为机器人坐标系
        robot_coordinates_action = clipped_scaled_action + self.default_dof_pos.unsqueeze(0)
        
        # 安全检查
        if not self.check_safety(robot_coordinates_action[0]):
            self.get_logger().warn("动作安全检查失败，跳过发送")
            return
        
        # 发送命令
        self._publish_legs_cmd(robot_coordinates_action[0], stand=False)
    
    def send_stand_action(self, actions: torch.Tensor):
        """发送站立动作"""
        actions = torch.tensor(actions, device=self.device).unsqueeze(0)
        self.actions = actions
        
        # 安全检查
        if not self.check_safety(actions[0]):
            self.get_logger().warn("站立动作安全检查失败，跳过发送")
            return
        
        self._publish_legs_cmd(actions[0], stand=True)
    
    def get_stand_action(self) -> List[float]:
        """获取站立动作"""
        if self.first_run:
            # TODO: 从实际的关节状态获取起始位置
            self.first_run = False
        
        self.percent_1 += 1.0 / self.duration_1
        self.percent_1 = min(self.percent_1, 1)
        
        if self.percent_1 < 1:
            for i in range(self.num_dof):
                self.stand_action[i] = (1 - self.percent_1) * self.start_pos[i] + self.percent_1 * self._target_pos_1[i]
            
            if self.first_run_target_1:
                self.get_logger().info('前往目标位置1')
                self.first_run_target_1 = False
        
        if (self.percent_1 == 1) and (self.percent_2 <= 1):
            self.percent_2 += 1.0 / self.duration_2
            self.percent_2 = min(self.percent_2, 1)
            for i in range(self.num_dof):
                self.stand_action[i] = (1 - self.percent_2) * self._target_pos_1[i] + self.percent_2 * self._target_pos_2[i]
        
        return self.stand_action
    
    def _publish_legs_cmd(self, robot_coordinates_action: torch.Tensor, stand: bool):
        """发布腿部命令"""
        # TODO: 实现具体的ROS2发布逻辑
        self.get_logger().debug(f"发送{'站立' if stand else '运动'}命令")
    
    def reset_obs(self):
        """重置观察"""
        self.start_pos = [0.0] * self.num_dof
        self.stand_action = [0.0] * self.num_dof
        
        self.percent_1 = 0
        self.percent_2 = 0
        
        self.first_run_target_1 = True
        self.first_run = True
        
        # 重置缓冲区
        self.actions = torch.zeros(self.num_actions, device=self.device, dtype=torch.float32)
        self.proprio_history_buf = torch.zeros(1, self.n_hist_len, self.n_proprio, device=self.device, dtype=torch.float)
        self.episode_length_buf = torch.zeros(1, device=self.device, dtype=torch.float)
        self.forward_depth_latent_yaw_buffer = torch.zeros(1, self.n_depth_latent + 2, device=self.device, dtype=torch.float)
        self.xyyaw_command = torch.tensor([[0, 0, 0]], device=self.device, dtype=torch.float32)
        self.contact_filt = torch.ones((1, 4), device=self.device, dtype=torch.float32)
        self.last_contact_filt = torch.ones((1, 4), device=self.device, dtype=torch.float32)
        
        # 重置推理引擎
        if self.inference_engine:
            self.inference_engine.reset()
        
        # 重置安全计数器
        self.safety_violations = 0
        
        # 重置计数器
        self.global_counter = 0
        
        self.get_logger().info("观察状态已重置")
    
    def print_performance_stats(self):
        """打印性能统计"""
        self.performance_timer.print_timings("本体感受处理: ")
        
        current_time = time.monotonic()
        if self.control_cycle_count > 0:
            avg_cycle_time = (current_time - self.last_control_time) / self.control_cycle_count
            self.get_logger().info(f"平均控制周期时间: {avg_cycle_time:.5f}s")
        
        self.get_logger().info(f"安全违规次数: {self.safety_violations}/{self.max_safety_violations}")
        self.get_logger().info(f"当前状态: {self.state_machine.get_current_state()}")
        
        # 打印推理引擎性能
        if self.inference_engine:
            self.inference_engine.print_performance_stats() 