#!/usr/bin/env python3
# test_motion_switcher.py
# 测试运动模式切换API

import rclpy
from rclpy.node import Node
import time
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from constants import MOTION_SWITCHER_API_ID_RELEASE, MOTION_SWITCHER_API_ID_SELECT_MCF
from ros_interface import ROSInterface
from robot_controller import RobotController
from robot_mode_manager import RobotModeManager


class MotionSwitcherTester(Node):
    """运动模式切换测试器"""
    
    def __init__(self):
        super().__init__("motion_switcher_tester")
        
        # 初始化组件
        self.robot_controller = RobotController(robot_name="go2_test")
        self.robot_mode_manager = RobotModeManager(self.robot_controller)
        self.ros_interface = ROSInterface(self, self.robot_controller, self.robot_mode_manager)
        
        # 设置ROS接口
        self.robot_controller.set_ros_interface(self.ros_interface)
        
        # 初始化ROS接口
        self.ros_interface.initialize()
        
        self.get_logger().info("运动模式切换测试器初始化完成")
    
    def test_motion_switcher_api(self):
        """测试运动模式切换API"""
        self.get_logger().info("开始测试运动模式切换API...")
        
        try:
            # 测试1: 切换到普通模式（释放运动模式）
            self.get_logger().info("测试1: 切换到普通模式")
            self.ros_interface.publish_motion_switcher(0)
            time.sleep(2)
            
            # 测试2: 切换到MCF模式（运动模式）
            self.get_logger().info("测试2: 切换到MCF模式")
            self.ros_interface.publish_motion_switcher(1)
            time.sleep(2)
            
            # 测试3: 再次切换到普通模式
            self.get_logger().info("测试3: 再次切换到普通模式")
            self.ros_interface.publish_motion_switcher(0)
            time.sleep(2)
            
            self.get_logger().info("运动模式切换API测试完成")
            
        except Exception as e:
            self.get_logger().error(f"测试过程中发生错误: {e}")
    
    def test_state_manager_integration(self):
        """测试状态管理器集成"""
        self.get_logger().info("开始测试状态管理器集成...")
        
        try:
            # 测试切换到运动模式
            self.get_logger().info("测试切换到运动模式")
            self.robot_mode_manager.switch_to_native_sport_mode()
            time.sleep(2)
            
            # 测试切换到站立策略
            self.get_logger().info("测试切换到站立策略")
            self.robot_mode_manager.switch_to_stand_policy_mode()
            time.sleep(2)
            
            # 测试切换到运动控制策略
            self.get_logger().info("测试切换到运动控制策略")
            self.robot_mode_manager.switch_to_locomotion_policy_mode()
            time.sleep(2)
            
            # 测试切换回运动模式
            self.get_logger().info("测试切换回运动模式")
            self.robot_mode_manager.switch_to_native_sport_mode()
            time.sleep(2)
            
            self.get_logger().info("状态管理器集成测试完成")
            
        except Exception as e:
            self.get_logger().error(f"状态管理器测试过程中发生错误: {e}")
    
    def test_robot_controller_integration(self):
        """测试机器人控制器集成"""
        self.get_logger().info("开始测试机器人控制器集成...")
        
        try:
            # 测试切换到运动模式
            self.get_logger().info("测试机器人控制器切换到运动模式")
            self.robot_controller.switch_to_sport_mode()
            time.sleep(2)
            
            # 测试切换到普通模式
            self.get_logger().info("测试机器人控制器切换到普通模式")
            self.robot_controller.switch_to_normal_mode()
            time.sleep(2)
            
            # 测试发布运动模式命令
            self.get_logger().info("测试发布运动模式命令")
            self.robot_controller.publish_sport_mode_command(1002)  # 平衡站立
            time.sleep(2)
            
            self.get_logger().info("机器人控制器集成测试完成")
            
        except Exception as e:
            self.get_logger().error(f"机器人控制器测试过程中发生错误: {e}")
    
    def run_all_tests(self):
        """运行所有测试"""
        self.get_logger().info("开始运行所有测试...")
        
        # 等待ROS接口准备就绪
        time.sleep(1)
        
        # 运行测试
        self.test_motion_switcher_api()
        time.sleep(1)
        
        self.test_state_manager_integration()
        time.sleep(1)
        
        self.test_robot_controller_integration()
        time.sleep(1)
        
        self.get_logger().info("所有测试完成")
    
    def print_api_info(self):
        """打印API信息"""
        print("\n=== 运动模式切换API信息 ===")
        print(f"释放模式 API ID: {MOTION_SWITCHER_API_ID_RELEASE}")
        print(f"选择MCF模式 API ID: {MOTION_SWITCHER_API_ID_SELECT_MCF}")
        print(f"话题: /api/motion_switcher/request")
        print("\n=== API参数 ===")
        print("功能\t\t\tAPI ID\t参数")
        print("关闭运动模式\t\t1003\t{}")
        print("开启运动模式\t\t1002\t{\"name\": \"mcf\"}")
        print("\n=== 使用说明 ===")
        print("1. 确保Go2软件版本为1.1.7或更高")
        print("2. 使用publish_motion_switcher()方法")
        print("3. 模式0 = 释放模式（普通模式）")
        print("4. 模式1 = 选择MCF模式（运动模式）")


def main():
    """主函数"""
    rclpy.init()
    
    # 创建测试器
    tester = MotionSwitcherTester()
    
    # 打印API信息
    tester.print_api_info()
    
    try:
        # 运行测试
        tester.run_all_tests()
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
    finally:
        # 清理
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main() 