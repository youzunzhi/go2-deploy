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
        self.state_machine = StateMachine("native_sport_mode")
        
        # 模式标志
        self.use_native_sport_mode = True
        self.use_stand_policy_mode = False
        self.use_locomotion_policy_mode = False
        
        # 计数器
        self.global_counter = 0
        self.visual_update_interval = ControlConfig.visual_update_interval
        
        # 回调函数 - Go2原生运动模式动作
        self.native_sport_action_callbacks: Dict[str, Callable] = {}
        # 回调函数 - 主控模式切换（native/stand_policy/locomotion_policy）
        self.main_mode_switch_callbacks: Dict[str, Callable] = {}
        
        # 性能监控
        self.mode_start_time = time.monotonic()
        self.mode_duration = 0.0
    
    def register_native_sport_action_callback(self, action: str, callback: Callable):
        """注册Go2原生运动模式动作回调"""
        self.native_sport_action_callbacks[action] = callback

    def register_main_mode_switch_callback(self, mode: str, callback: Callable):
        """注册主控模式切换回调（native/stand_policy/locomotion_policy）"""
        self.main_mode_switch_callbacks[mode] = callback
    
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
        if self.use_native_sport_mode:
            return self._handle_native_sport_mode_input(joy_stick_buffer)
        
        # 其他模式下的手柄处理
        return self._handle_other_mode_input(joy_stick_buffer)
    
    def _handle_native_sport_mode_input(self, joy_stick_buffer) -> bool:
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
            self.switch_to_stand_policy_mode()
            handled = True
        
        return handled
    
    def _handle_other_mode_input(self, joy_stick_buffer) -> bool:
        """处理其他模式下的手柄输入"""
        handled = False
        
        # Y: 切换到运动控制策略
        if (joy_stick_buffer.keys & WirelessButtons.Y):
            self.switch_to_locomotion_policy_mode()
            handled = True
        
        # L2: 切换回运动模式
        elif (joy_stick_buffer.keys & WirelessButtons.L2):
            self.switch_to_native_sport_mode()
            handled = True
        
        return handled
    
    def _execute_sport_mode_callback(self, mode: str):
        """执行运动模式回调"""
        if mode in self.native_sport_action_callbacks:
            try:
                self.native_sport_action_callbacks[mode]()
            except Exception as e:
                print(f"执行运动模式回调失败: {e}")
    
    def switch_to_native_sport_mode(self):
        """切换到Go2原生运动主模式"""
        if self.state_machine.transition_to("native_sport_mode"):
            self.use_native_sport_mode = True
            self.use_stand_policy_mode = False
            self.use_locomotion_policy_mode = False
            self.global_counter = 0
            self._execute_main_mode_switch_callback("native_sport_mode")
            print("切换到Go2原生运动主模式")
    
    def switch_to_stand_policy_mode(self):
        """切换到自定义站立主模式"""
        if self.state_machine.transition_to("stand_policy_mode"):
            self.use_native_sport_mode = False
            self.use_stand_policy_mode = True
            self.use_locomotion_policy_mode = False
            self._execute_main_mode_switch_callback("stand_policy_mode")
            print("切换到自定义站立主模式")
    
    def switch_to_locomotion_policy_mode(self):
        """切换到自定义locomotion主模式"""
        if self.state_machine.transition_to("locomotion_policy_mode"):
            self.use_native_sport_mode = False
            self.use_stand_policy_mode = False
            self.use_locomotion_policy_mode = True
            self.global_counter = 0
            self._execute_main_mode_switch_callback("locomotion_policy_mode")
            print("切换到自定义locomotion主模式")
    
    def _execute_main_mode_switch_callback(self, mode: str):
        """执行主控模式切换回调"""
        if mode in self.main_mode_switch_callbacks:
            try:
                self.main_mode_switch_callbacks[mode]()
            except Exception as e:
                print(f"执行主控模式切换回调失败: {e}")
    
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
            "use_native_sport_mode": self.use_native_sport_mode,
            "use_stand_policy_mode": self.use_stand_policy_mode,
            "use_locomotion_policy_mode": self.use_locomotion_policy_mode,
            "global_counter": self.global_counter,
            "mode_duration": self.mode_duration,
            "visual_update_interval": self.visual_update_interval
        }
    
    def reset(self):
        """重置状态管理器"""
        self.state_machine = StateMachine("native_sport_mode")
        self.use_native_sport_mode = True
        self.use_stand_policy_mode = False
        self.use_locomotion_policy_mode = False
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
        # Go2原生运动模式动作回调
        self.state_manager.register_native_sport_action_callback("standup", self._sport_standup)
        self.state_manager.register_native_sport_action_callback("standdown", self._sport_standdown)
        self.state_manager.register_native_sport_action_callback("balancestand", self._sport_balancestand)
        # 主控模式切换回调
        self.state_manager.register_main_mode_switch_callback("native_sport_mode", self._on_native_sport_mode)
        self.state_manager.register_main_mode_switch_callback("stand_policy_mode", self._on_stand_policy_mode)
        self.state_manager.register_main_mode_switch_callback("locomotion_policy_mode", self._on_locomotion_policy_mode)
    
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
    
    def _on_native_sport_mode(self):
        """进入运动模式（统一处理硬件切换和应用逻辑）"""
        print("进入运动模式")
        
        # 硬件切换：切换到MCF模式
        if hasattr(self.robot_controller, 'ros_interface'):
            print("切换到MCF模式（运动模式）")
            self.robot_controller.ros_interface.publish_motion_switcher(1)
        
        # 应用逻辑：重置观察
        self.robot_controller.reset_obs()
        # TODO: 实现其他运动模式初始化
    
    def _on_stand_policy_mode(self):
        """进入站立策略（统一处理硬件切换和应用逻辑）"""
        print("进入站立策略")
        
        # 硬件切换：释放运动模式
        if hasattr(self.robot_controller, 'ros_interface'):
            print("切换到普通模式（释放运动模式）")
            self.robot_controller.ros_interface.publish_motion_switcher(0)
        
        # TODO: 实现站立策略初始化
    
    def _on_locomotion_policy_mode(self):
        """进入运动控制策略（统一处理硬件切换和应用逻辑）"""
        print("进入运动控制策略")
        
        # 硬件切换：释放运动模式
        if hasattr(self.robot_controller, 'ros_interface'):
            print("切换到普通模式（释放运动模式）")
            self.robot_controller.ros_interface.publish_motion_switcher(0)
        
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
        if self.state_manager.use_native_sport_mode:
            self._execute_native_sport_mode()
        elif self.state_manager.use_stand_policy_mode:
            self._execute_stand_policy_mode()
        elif self.state_manager.use_locomotion_policy_mode:
            self._execute_locomotion_policy_mode()
    
    def _execute_native_sport_mode(self):
        """执行运动模式"""
        # 运动模式通常由手柄直接控制，这里不需要额外的执行逻辑
        pass
    
    def _execute_stand_policy_mode(self):
        """执行站立策略"""
        stand_action = self.robot_controller.get_stand_action()
        self.robot_controller.send_stand_action(stand_action)
    
    def _execute_locomotion_policy_mode(self):
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
            self.state_manager.switch_to_native_sport_mode()
    
    def get_mode_info(self) -> Dict[str, Any]:
        """获取模式信息"""
        return self.state_manager.get_state_info() 