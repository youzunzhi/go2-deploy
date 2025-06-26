# ros_interface.py
# ROS2接口模块

import rclpy
from rclpy.node import Node
import torch
import numpy as np
from typing import Optional, Callable, Any

# 注意：这些导入需要根据实际的ROS2包进行调整
# from unitree_go.msg import WirelessController, LowState, LowCmd
# from unitree_api.msg import Request
# from std_msgs.msg import Float32MultiArray


class ROSInterface:
    """ROS2接口类"""
    
    def __init__(self, node: Node, robot_controller, state_manager):
        """
        初始化ROS接口
        
        Args:
            node: ROS2节点
            robot_controller: 机器人控制器
            state_manager: 状态管理器
        """
        self.node = node
        self.robot_controller = robot_controller
        self.state_manager = state_manager
        
        # 话题名称
        self.low_state_topic = "/lowstate"
        self.low_cmd_topic = "/lowcmd"
        self.joy_stick_topic = "/wirelesscontroller"
        self.depth_data_topic = "/forward_depth_image"
        self.sport_state_topic = "/api/robot_state/request"
        self.sport_mode_topic = "/api/sport/request"
        
        # 缓冲区
        self.low_state_buffer = None
        self.joy_stick_buffer = None
        self.depth_data = None
        
        # 发布者和订阅者
        self.low_cmd_pub = None
        self.low_state_sub = None
        self.joy_stick_sub = None
        self.depth_input_sub = None
        self.sport_state_pub = None
        self.sport_mode_pub = None
        
        # 回调函数
        self.state_callbacks: list[Callable] = []
        self.joystick_callbacks: list[Callable] = []
        self.depth_callbacks: list[Callable] = []
        
        # 初始化标志
        self.initialized = False
        
        self.node.get_logger().info("ROS接口初始化完成")
    
    def initialize(self):
        """初始化ROS接口"""
        try:
            self._create_publishers()
            self._create_subscribers()
            self.initialized = True
            self.node.get_logger().info("ROS接口初始化成功")
        except Exception as e:
            self.node.get_logger().error(f"ROS接口初始化失败: {e}")
            self.initialized = False
    
    def _create_publishers(self):
        """创建发布者"""
        # 注意：这里需要根据实际的ROS2消息类型进行调整
        # self.low_cmd_pub = self.node.create_publisher(
        #     LowCmd, self.low_cmd_topic, 1
        # )
        # self.sport_state_pub = self.node.create_publisher(
        #     Request, self.sport_state_topic, 1
        # )
        # self.sport_mode_pub = self.node.create_publisher(
        #     Request, self.sport_mode_topic, 1
        # )
        
        self.node.get_logger().info("ROS发布者创建完成")
    
    def _create_subscribers(self):
        """创建订阅者"""
        # 注意：这里需要根据实际的ROS2消息类型进行调整
        # self.low_state_sub = self.node.create_subscription(
        #     LowState, self.low_state_topic, self._low_state_callback, 1
        # )
        # self.joy_stick_sub = self.node.create_subscription(
        #     WirelessController, self.joy_stick_topic, self._joy_stick_callback, 1
        # )
        # self.depth_input_sub = self.node.create_subscription(
        #     Float32MultiArray, self.depth_data_topic, self._depth_data_callback, 1
        # )
        
        self.node.get_logger().info("ROS订阅者创建完成")
    
    def register_state_callback(self, callback: Callable):
        """注册状态回调"""
        self.state_callbacks.append(callback)
    
    def register_joystick_callback(self, callback: Callable):
        """注册手柄回调"""
        self.joystick_callbacks.append(callback)
    
    def register_depth_callback(self, callback: Callable):
        """注册深度数据回调"""
        self.depth_callbacks.append(callback)
    
    def _low_state_callback(self, msg):
        """低层状态回调"""
        self.low_state_buffer = msg
        
        # 更新机器人控制器的关节状态
        self._update_joint_states(msg)
        
        # 执行状态回调
        for callback in self.state_callbacks:
            try:
                callback(msg)
            except Exception as e:
                self.node.get_logger().error(f"状态回调执行失败: {e}")
    
    def _update_joint_states(self, msg):
        """更新关节状态"""
        try:
            # 注意：这里需要根据实际的ROS2消息结构进行调整
            # for sim_idx in range(self.robot_controller.num_dof):
            #     real_idx = self.robot_controller.dof_map[sim_idx]
            #     self.robot_controller.dof_pos_[0, sim_idx] = (
            #         msg.motor_state[real_idx].q * self.robot_controller.dof_signs[sim_idx]
            #     )
            #     self.robot_controller.dof_vel_[0, sim_idx] = (
            #         msg.motor_state[real_idx].dq * self.robot_controller.dof_signs[sim_idx]
            #     )
            pass
        except Exception as e:
            self.node.get_logger().error(f"更新关节状态失败: {e}")
    
    def _joy_stick_callback(self, msg):
        """手柄回调"""
        self.joy_stick_buffer = msg
        
        # 更新命令
        self._update_commands(msg)
        
        # 执行手柄回调
        for callback in self.joystick_callbacks:
            try:
                callback(msg)
            except Exception as e:
                self.node.get_logger().error(f"手柄回调执行失败: {e}")
    
    def _update_commands(self, msg):
        """更新命令"""
        try:
            # 注意：这里需要根据实际的ROS2消息结构进行调整
            # 处理手柄输入并更新xyyaw_command
            # ly = msg.ly
            # lx = msg.lx
            # rx = msg.rx
            # 
            # # 计算vx, vy, yaw
            # vx = self._process_joystick_axis(ly, self.robot_controller.lin_vel_deadband)
            # yaw = self._process_joystick_axis(-lx, self.robot_controller.ang_vel_deadband)
            # vy = self._process_joystick_axis(-rx, self.robot_controller.lin_vel_deadband)
            # 
            # self.robot_controller.xyyaw_command = torch.tensor(
            #     [[vx, vy, yaw]], device=self.robot_controller.device, dtype=torch.float32
            # )
            pass
        except Exception as e:
            self.node.get_logger().error(f"更新命令失败: {e}")
    
    def _depth_data_callback(self, msg):
        """深度数据回调"""
        try:
            # 注意：这里需要根据实际的ROS2消息结构进行调整
            # self.depth_data = torch.tensor(
            #     msg.data, dtype=torch.float32
            # ).reshape(1, 58, 87).to(self.robot_controller.device)
            # 
            # self.robot_controller.depth_data = self.depth_data
            pass
        except Exception as e:
            self.node.get_logger().error(f"处理深度数据失败: {e}")
        
        # 执行深度数据回调
        for callback in self.depth_callbacks:
            try:
                callback(self.depth_data)
            except Exception as e:
                self.node.get_logger().error(f"深度数据回调执行失败: {e}")
    
    def publish_low_cmd(self, motor_commands):
        """发布低层命令"""
        if not self.initialized or self.low_cmd_pub is None:
            return
        
        try:
            # 注意：这里需要根据实际的ROS2消息结构进行调整
            # msg = LowCmd()
            # 
            # for sim_idx in range(self.robot_controller.num_dof):
            #     real_idx = self.robot_controller.dof_map[sim_idx]
            #     if not self.robot_controller.dryrun:
            #         msg.motor_cmd[real_idx].mode = self.robot_controller.turn_on_motor_mode[sim_idx]
            #     msg.motor_cmd[real_idx].q = motor_commands[sim_idx].item() * self.robot_controller.dof_signs[sim_idx]
            #     msg.motor_cmd[real_idx].dq = 0.0
            #     msg.motor_cmd[real_idx].tau = 0.0
            #     msg.motor_cmd[real_idx].kp = self.robot_controller.p_gains[sim_idx].item()
            #     msg.motor_cmd[real_idx].kd = self.robot_controller.d_gains[sim_idx].item()
            # 
            # msg.crc = get_crc(msg)
            # self.low_cmd_pub.publish(msg)
            pass
        except Exception as e:
            self.node.get_logger().error(f"发布低层命令失败: {e}")
    
    def publish_sport_mode(self, mode_id: int):
        """发布运动模式命令"""
        if not self.initialized or self.sport_mode_pub is None:
            return
        
        try:
            # 注意：这里需要根据实际的ROS2消息结构进行调整
            # msg = Request()
            # msg.header.identity.id = 0
            # msg.header.identity.api_id = mode_id
            # msg.header.lease.id = 0
            # msg.header.policy.priority = 0
            # msg.header.policy.noreply = False
            # msg.parameter = ''
            # msg.binary = []
            # 
            # self.sport_mode_pub.publish(msg)
            pass
        except Exception as e:
            self.node.get_logger().error(f"发布运动模式命令失败: {e}")
    
    def publish_sport_state(self, state: int):
        """发布运动状态命令"""
        if not self.initialized or self.sport_state_pub is None:
            return
        
        try:
            # 注意：这里需要根据实际的ROS2消息结构进行调整
            # msg = Request()
            # msg.header.identity.id = 0
            # msg.header.identity.api_id = 1001
            # msg.header.lease.id = 0
            # msg.header.policy.priority = 0
            # msg.header.policy.noreply = False
            # 
            # if state == 0:
            #     msg.parameter = '{"name":"sport_mode","switch":0}'
            # elif state == 1:
            #     msg.parameter = '{"name":"sport_mode","switch":1}'
            # 
            # msg.binary = []
            # self.sport_state_pub.publish(msg)
            pass
        except Exception as e:
            self.node.get_logger().error(f"发布运动状态命令失败: {e}")
    
    def wait_for_messages(self, timeout: float = 10.0) -> bool:
        """
        等待接收关键消息
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            bool: 是否成功接收到消息
        """
        import time
        
        start_time = time.monotonic()
        self.node.get_logger().info("等待接收关键消息...")
        
        while time.monotonic() - start_time < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            
            if (self.low_state_buffer is not None and 
                self.joy_stick_buffer is not None):
                self.node.get_logger().info("关键消息接收完成")
                return True
        
        self.node.get_logger().warn("等待消息超时")
        return False
    
    def is_ready(self) -> bool:
        """检查是否准备就绪"""
        return (self.initialized and 
                self.low_state_buffer is not None and 
                self.joy_stick_buffer is not None)
    
    def get_status(self) -> dict:
        """获取状态信息"""
        return {
            "initialized": self.initialized,
            "low_state_received": self.low_state_buffer is not None,
            "joystick_received": self.joy_stick_buffer is not None,
            "depth_data_received": self.depth_data is not None,
            "ready": self.is_ready()
        }


class MockROSInterface(ROSInterface):
    """模拟ROS接口，用于测试"""
    
    def __init__(self, node: Node, robot_controller, state_manager):
        super().__init__(node, robot_controller, state_manager)
        
        # 模拟数据
        self._mock_low_state()
        self._mock_joystick()
        self._mock_depth_data()
    
    def _mock_low_state(self):
        """模拟低层状态数据"""
        # 创建模拟的低层状态消息
        # 这里需要根据实际的ROS2消息结构进行调整
        pass
    
    def _mock_joystick(self):
        """模拟手柄数据"""
        # 创建模拟的手柄消息
        # 这里需要根据实际的ROS2消息结构进行调整
        pass
    
    def _mock_depth_data(self):
        """模拟深度数据"""
        # 创建模拟的深度数据
        self.depth_data = torch.randn(1, 58, 87, device=self.robot_controller.device)
        self.robot_controller.depth_data = self.depth_data
    
    def wait_for_messages(self, timeout: float = 1.0) -> bool:
        """模拟等待消息"""
        import time
        time.sleep(0.1)  # 模拟短暂延迟
        return True 