# state_manager.py
# 状态管理器

import time
from typing import Optional, Callable, Dict, Any
from constants import *
from utils import StateMachine


class RobotStateManager:
    """机器人状态管理器"""
    
    def __init__(self):
        """初始化状态管理器"""
        # 状态机
        self.state_machine = StateMachine("sport_mode")
        
        # 模式标志
        self.use_stand_policy = False
        self.use_locomotion_policy = False
        self.use_sport_mode = True
        
        # 计数器
        self.global_counter = 0
        self.visual_update_interval = ControlConfig.visual_update_interval
        
        # 回调函数
        self.sport_mode_callbacks: Dict[str, Callable] = {}
        self.state_change_callbacks: Dict[str, Callable] = {}
        
        # 性能监控
        self.mode_start_time = time.monotonic()
        self.mode_duration = 0.0
    
    def register_sport_mode_callback(self, mode: str, callback: Callable):
        """注册运动模式回调"""
        self.sport_mode_callbacks[mode] = callback
    
    def register_state_change_callback(self, state: str, callback: Callable):
        """注册状态变化回调"""
        self.state_change_callbacks[state] = callback
    
    def handle_joystick_input(self, joy_stick_buffer) -> bool:
        """
        处理手柄输入
        
        Args:
            joy_stick_buffer: 手柄缓冲区
            
        Returns:
            bool: 是否处理了输入
        """
        if not hasattr(joy_stick_buffer, 'keys'):
            return False
        
        # 运动模式下的手柄处理
        if self.use_sport_mode:
            return self._handle_sport_mode_input(joy_stick_buffer)
        
        # 其他模式下的手柄处理
        return self._handle_other_mode_input(joy_stick_buffer)
    
    def _handle_sport_mode_input(self, joy_stick_buffer) -> bool:
        """处理运动模式下的手柄输入"""
        handled = False
        
        # R1: 站立
        if (joy_stick_buffer.keys & WirelessButtons.R1):
            self._execute_sport_mode_callback("standup")
            handled = True
        
        # R2: 坐下
        elif (joy_stick_buffer.keys & WirelessButtons.R2):
            self._execute_sport_mode_callback("standdown")
            handled = True
        
        # X: 平衡站立
        elif (joy_stick_buffer.keys & WirelessButtons.X):
            self._execute_sport_mode_callback("balancestand")
            handled = True
        
        # L1: 切换到站立策略
        elif (joy_stick_buffer.keys & WirelessButtons.L1):
            self.switch_to_stand_policy()
            handled = True
        
        return handled
    
    def _handle_other_mode_input(self, joy_stick_buffer) -> bool:
        """处理其他模式下的手柄输入"""
        handled = False
        
        # Y: 切换到运动控制策略
        if (joy_stick_buffer.keys & WirelessButtons.Y):
            self.switch_to_locomotion_policy()
            handled = True
        
        # L2: 切换回运动模式
        elif (joy_stick_buffer.keys & WirelessButtons.L2):
            self.switch_to_sport_mode()
            handled = True
        
        return handled
    
    def _execute_sport_mode_callback(self, mode: str):
        """执行运动模式回调"""
        if mode in self.sport_mode_callbacks:
            try:
                self.sport_mode_callbacks[mode]()
            except Exception as e:
                print(f"执行运动模式回调失败: {e}")
    
    def switch_to_sport_mode(self):
        """切换到运动模式"""
        if self.state_machine.transition_to("sport_mode"):
            self.use_sport_mode = True
            self.use_stand_policy = False
            self.use_locomotion_policy = False
            
            # 重置计数器
            self.global_counter = 0
            
            # 执行状态变化回调
            self._execute_state_change_callback("sport_mode")
            
            print("切换到运动模式")
    
    def switch_to_stand_policy(self):
        """切换到站立策略"""
        if self.state_machine.transition_to("stand_policy"):
            self.use_sport_mode = False
            self.use_stand_policy = True
            self.use_locomotion_policy = False
            
            # 执行状态变化回调
            self._execute_state_change_callback("stand_policy")
            
            print("切换到站立策略")
    
    def switch_to_locomotion_policy(self):
        """切换到运动控制策略"""
        if self.state_machine.transition_to("locomotion_policy"):
            self.use_sport_mode = False
            self.use_stand_policy = False
            self.use_locomotion_policy = True
            
            # 重置计数器
            self.global_counter = 0
            
            # 执行状态变化回调
            self._execute_state_change_callback("locomotion_policy")
            
            print("切换到运动控制策略")
    
    def _execute_state_change_callback(self, state: str):
        """执行状态变化回调"""
        if state in self.state_change_callbacks:
            try:
                self.state_change_callbacks[state]()
            except Exception as e:
                print(f"执行状态变化回调失败: {e}")
    
    def update_counter(self):
        """更新计数器"""
        self.global_counter += 1
    
    def should_update_vision(self) -> bool:
        """是否应该更新视觉"""
        return self.global_counter % self.visual_update_interval == 0
    
    def get_current_state(self) -> str:
        """获取当前状态"""
        return self.state_machine.get_current_state()
    
    def get_state_info(self) -> Dict[str, Any]:
        """获取状态信息"""
        current_time = time.monotonic()
        self.mode_duration = current_time - self.mode_start_time
        
        return {
            "current_state": self.get_current_state(),
            "use_sport_mode": self.use_sport_mode,
            "use_stand_policy": self.use_stand_policy,
            "use_locomotion_policy": self.use_locomotion_policy,
            "global_counter": self.global_counter,
            "mode_duration": self.mode_duration,
            "visual_update_interval": self.visual_update_interval
        }
    
    def reset(self):
        """重置状态管理器"""
        self.state_machine = StateMachine("sport_mode")
        self.use_stand_policy = False
        self.use_locomotion_policy = False
        self.use_sport_mode = True
        self.global_counter = 0
        self.mode_start_time = time.monotonic()
        self.mode_duration = 0.0
        
        print("状态管理器已重置")


class ModeController:
    """模式控制器"""
    
    def __init__(self, robot_controller, state_manager):
        """
        初始化模式控制器
        
        Args:
            robot_controller: 机器人控制器
            state_manager: 状态管理器
        """
        self.robot_controller = robot_controller
        self.state_manager = state_manager
        
        # 注册回调
        self._register_callbacks()
    
    def _register_callbacks(self):
        """注册回调函数"""
        # 运动模式回调
        self.state_manager.register_sport_mode_callback("standup", self._sport_standup)
        self.state_manager.register_sport_mode_callback("standdown", self._sport_standdown)
        self.state_manager.register_sport_mode_callback("balancestand", self._sport_balancestand)
        
        # 状态变化回调
        self.state_manager.register_state_change_callback("sport_mode", self._on_sport_mode)
        self.state_manager.register_state_change_callback("stand_policy", self._on_stand_policy)
        self.state_manager.register_state_change_callback("locomotion_policy", self._on_locomotion_policy)
    
    def _sport_standup(self):
        """运动模式：站立"""
        print("执行运动模式：站立")
        # TODO: 实现具体的站立命令
    
    def _sport_standdown(self):
        """运动模式：坐下"""
        print("执行运动模式：坐下")
        # TODO: 实现具体的坐下命令
    
    def _sport_balancestand(self):
        """运动模式：平衡站立"""
        print("执行运动模式：平衡站立")
        # TODO: 实现具体的平衡站立命令
    
    def _on_sport_mode(self):
        """进入运动模式"""
        print("进入运动模式")
        self.robot_controller.reset_obs()
        # TODO: 实现运动模式初始化
    
    def _on_stand_policy(self):
        """进入站立策略"""
        print("进入站立策略")
        # TODO: 实现站立策略初始化
    
    def _on_locomotion_policy(self):
        """进入运动控制策略"""
        print("进入运动控制策略")
        # TODO: 实现运动控制策略初始化
    
    def execute_current_mode(self, joy_stick_buffer):
        """
        执行当前模式
        
        Args:
            joy_stick_buffer: 手柄缓冲区
        """
        # 处理手柄输入
        self.state_manager.handle_joystick_input(joy_stick_buffer)
        
        # 执行当前模式
        if self.state_manager.use_sport_mode:
            self._execute_sport_mode()
        elif self.state_manager.use_stand_policy:
            self._execute_stand_policy()
        elif self.state_manager.use_locomotion_policy:
            self._execute_locomotion_policy()
    
    def _execute_sport_mode(self):
        """执行运动模式"""
        # 运动模式通常由手柄直接控制，这里不需要额外的执行逻辑
        pass
    
    def _execute_stand_policy(self):
        """执行站立策略"""
        stand_action = self.robot_controller.get_stand_action()
        self.robot_controller.send_stand_action(stand_action)
    
    def _execute_locomotion_policy(self):
        """执行运动控制策略"""
        try:
            # 执行推理
            action = self.robot_controller.execute_locomotion_policy()
            
            # 发送动作
            self.robot_controller.send_action(action)
            
            # 更新计数器
            self.state_manager.update_counter()
            
        except Exception as e:
            print(f"运动控制策略执行失败: {e}")
            # 发生异常时切换到运动模式
            self.state_manager.switch_to_sport_mode()
    
    def get_mode_info(self) -> Dict[str, Any]:
        """获取模式信息"""
        return self.state_manager.get_state_info() 