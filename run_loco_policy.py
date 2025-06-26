#!/usr/bin/env python3
# run_loco_policy.py
# 主运行脚本 - 运动控制策略部署

import rclpy
from rclpy.node import Node
import argparse
import os
import sys
import time
import signal
import threading
from typing import Optional

# 导入项目模块
from constants import RunMode
from config import RobotConfiguration, DeploymentConfig
from robot_controller import RobotController, WirelessButtons
from state_manager import RobotStateManager, ModeController
from ros_interface import ROSInterface
from inference_engine import InferenceEngine
from logger import SystemLogger, HealthMonitor


class LocomotionPolicyRunner(Node):
    """运动控制策略运行器"""
    
    def __init__(self, args):
        """
        初始化运行器
        
        Args:
            args: 命令行参数
        """
        super().__init__("locomotion_policy_runner")
        
        # 解析参数
        self.args = args
        self._parse_args()
        
        # 配置管理
        self.config = RobotConfiguration()
        self.deploy_config = DeploymentConfig()
        self.deploy_config.from_args(args)
        
        # 初始化日志系统
        self.system_logger = SystemLogger(
            log_dir=self.deploy_config.logdir,
            log_level="INFO"
        )
        self.health_monitor = HealthMonitor(self.system_logger)
        
        # 打印配置
        self.deploy_config.print_config()
        
        # 初始化组件
        self._init_components()
        
        # 控制循环状态
        self.is_running = False
        self.control_thread = None
        self.control_interval = self.deploy_config.duration
        
        # 性能监控
        self.cycle_count = 0
        self.start_time = time.monotonic()
        self.last_cycle_time = time.monotonic()
        
        # 信号处理
        self._setup_signal_handlers()
        
        self.system_logger.info("运动控制策略运行器初始化完成")
    
    def _parse_args(self):
        """解析命令行参数"""
        if not hasattr(self.args, 'model_dir'):
            self.args.model_dir = None
        if not hasattr(self.args, 'device'):
            self.args.device = "cuda"
        if not hasattr(self.args, 'duration'):
            self.args.duration = 0.02
        if not hasattr(self.args, 'dryrun'):
            self.args.dryrun = True
        if not hasattr(self.args, 'mode'):
            self.args.mode = RunMode.LOCOMOTION
        if not hasattr(self.args, 'loop_mode'):
            self.args.loop_mode = "timer"
    
    def _init_components(self):
        """初始化组件"""
        try:
            # 初始化机器人控制器
            self.robot_controller = RobotController(
                robot_name="go2",
                config=self.config,
                deploy_config=self.deploy_config
            )
            self.system_logger.info("机器人控制器初始化完成")
            
            # 初始化状态管理器
            self.state_manager = RobotStateManager()
            self.system_logger.info("状态管理器初始化完成")
            
            # 初始化模式控制器
            self.mode_controller = ModeController(
                self.robot_controller, 
                self.state_manager
            )
            self.system_logger.info("模式控制器初始化完成")
            
            # 初始化ROS接口
            self.ros_interface = ROSInterface(self.robot_controller)
            self.system_logger.info("ROS接口初始化完成")
            
            # 初始化推理引擎（如果配置了模型目录）
            if self.deploy_config.model_dir:
                self.inference_engine = InferenceEngine(
                    self.deploy_config.model_dir,
                    self.deploy_config.device,
                    self.deploy_config.warm_up_iterations
                )
                self.system_logger.info("推理引擎初始化完成")
            else:
                self.inference_engine = None
                self.system_logger.warning("未配置模型目录，推理引擎未初始化")
            
        except Exception as e:
            self.system_logger.error("组件初始化失败", e)
            raise
    
    def _setup_signal_handlers(self):
        """设置信号处理器"""
        def signal_handler(signum, frame):
            self.system_logger.info(f"收到信号 {signum}，开始优雅关闭...")
            self.shutdown()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def start(self):
        """启动运行器"""
        try:
            self.system_logger.info("启动运动控制策略运行器...")
            
            # 初始化推理引擎
            if self.inference_engine and not self.inference_engine.initialize():
                self.system_logger.error("推理引擎初始化失败")
                return False
            
            # 启动ROS接口
            self.ros_interface.start()
            
            # 启动控制循环
            if self.deploy_config.loop_mode == "timer":
                self._start_timer_loop()
            else:
                self._start_while_loop()
            
            self.is_running = True
            self.system_logger.info("运动控制策略运行器启动成功")
            
            return True
            
        except Exception as e:
            self.system_logger.error("启动失败", e)
            return False
    
    def _start_timer_loop(self):
        """启动定时器循环"""
        self.system_logger.info("使用定时器模式启动控制循环")
        
        # 创建定时器
        self.control_timer = self.create_timer(
            self.control_interval, 
            self._control_loop_callback
        )
    
    def _start_while_loop(self):
        """启动while循环"""
        self.system_logger.info("使用while循环模式启动控制循环")
        
        # 创建控制线程
        self.control_thread = threading.Thread(target=self._control_loop_thread)
        self.control_thread.daemon = True
        self.control_thread.start()
    
    def _control_loop_thread(self):
        """控制循环线程"""
        self.system_logger.info("控制循环线程启动")
        
        while self.is_running and rclpy.ok():
            loop_start_time = time.monotonic()
            
            try:
                self._execute_control_step()
            except Exception as e:
                self.system_logger.error("控制步骤执行失败", e)
                self.system_logger.performance_monitor.increment_metric("error_count")
            
            # 控制循环频率
            elapsed_time = time.monotonic() - loop_start_time
            sleep_time = max(0, self.control_interval - elapsed_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _control_loop_callback(self):
        """定时器回调函数"""
        try:
            self._execute_control_step()
        except Exception as e:
            self.system_logger.error("控制步骤执行失败", e)
            self.system_logger.performance_monitor.increment_metric("error_count")
    
    def _execute_control_step(self):
        """执行控制步骤"""
        cycle_start_time = time.monotonic()
        
        try:
            # 获取手柄输入
            joy_stick_buffer = self.ros_interface.get_joystick_buffer()
            
            # 执行当前模式
            self.mode_controller.execute_current_mode(joy_stick_buffer)
            
            # 更新性能统计
            self.cycle_count += 1
            cycle_time = time.monotonic() - cycle_start_time
            
            # 记录性能指标
            self._record_performance_metrics(cycle_time)
            
            # 检查系统健康状态
            self._check_system_health()
            
            # 定期打印统计信息
            if self.cycle_count % 1000 == 0:
                self._print_performance_stats()
            
        except Exception as e:
            self.system_logger.error("控制步骤执行异常", e)
            self.system_logger.performance_monitor.increment_metric("error_count")
    
    def _record_performance_metrics(self, cycle_time: float):
        """记录性能指标"""
        # 更新循环时间
        self.system_logger.performance_monitor.update_metric("control_cycle_time", cycle_time)
        self.system_logger.performance_monitor.increment_metric("cycle_count")
        
        # 记录推理引擎性能
        if self.inference_engine:
            inference_stats = self.inference_engine.get_performance_stats()
            if inference_stats["inference_count"] > 0:
                self.system_logger.performance_monitor.update_metric(
                    "inference_time", 
                    inference_stats["avg_inference_time"]
                )
        
        # 记录机器人控制器性能
        self.system_logger.performance_monitor.update_metric(
            "proprio_time", 
            cycle_time * 0.3  # 估算本体感受处理时间
        )
        
        # 记录动作时间
        self.system_logger.performance_monitor.update_metric(
            "action_time", 
            cycle_time * 0.2  # 估算动作发送时间
        )
        
        # 记录安全违规
        safety_violations = self.robot_controller.safety_violations
        self.system_logger.performance_monitor.update_metric("safety_violations", safety_violations)
        
        # 记录模式变化
        mode_info = self.mode_controller.get_mode_info()
        if mode_info["mode_duration"] < 1.0:  # 新切换的模式
            self.system_logger.performance_monitor.increment_metric("mode_changes")
    
    def _check_system_health(self):
        """检查系统健康状态"""
        metrics = self.system_logger.performance_monitor.get_metrics()
        health_status = self.health_monitor.check_system_health(metrics)
        
        # 如果系统状态异常，记录警告
        if health_status["system"] != "healthy":
            self.system_logger.warning(f"系统健康状态异常: {health_status['system']}")
    
    def _print_performance_stats(self):
        """打印性能统计"""
        current_time = time.monotonic()
        runtime = current_time - self.start_time
        
        print(f"\n=== 性能统计 (运行时间: {runtime:.1f}s) ===")
        print(f"控制循环次数: {self.cycle_count}")
        print(f"平均循环频率: {self.cycle_count / runtime:.1f} Hz")
        
        # 打印系统日志器性能摘要
        self.system_logger.print_performance_summary()
        
        # 打印健康状态
        self.health_monitor.print_health_summary()
        
        # 打印机器人控制器性能
        self.robot_controller.print_performance_stats()
        
        # 打印推理引擎性能
        if self.inference_engine:
            self.inference_engine.print_performance_stats()
        
        # 打印模式信息
        mode_info = self.mode_controller.get_mode_info()
        print(f"当前模式: {mode_info['current_state']}")
        print(f"模式持续时间: {mode_info['mode_duration']:.1f}s")
        print(f"全局计数器: {mode_info['global_counter']}")
        print("=" * 50)
    
    def shutdown(self):
        """关闭运行器"""
        self.system_logger.info("开始关闭运动控制策略运行器...")
        
        # 停止控制循环
        self.is_running = False
        
        # 停止定时器
        if hasattr(self, 'control_timer'):
            self.control_timer.cancel()
        
        # 等待控制线程结束
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=2.0)
        
        # 关闭ROS接口
        if hasattr(self, 'ros_interface'):
            self.ros_interface.shutdown()
        
        # 清理推理引擎
        if hasattr(self, 'inference_engine') and self.inference_engine:
            self.inference_engine.cleanup()
        
        # 清理机器人控制器
        if hasattr(self, 'robot_controller'):
            self.robot_controller.cleanup()
        
        # 保存日志和报告
        self.system_logger.save_error_report()
        self.system_logger.log_performance({})  # 保存最终性能数据
        
        # 打印最终统计
        self._print_performance_stats()
        
        self.system_logger.info("运动控制策略运行器已关闭")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="GO2运动控制策略部署")
    
    parser.add_argument(
        "--model_dir", 
        type=str, 
        default=None,
        help="模型目录路径，包含base_jit.pt和vision_weight.pt文件"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        choices=["cuda", "cpu"],
        help="推理设备 (默认: cuda)"
    )
    
    parser.add_argument(
        "--duration", 
        type=float, 
        default=0.02,
        help="控制周期，单位秒 (默认: 0.02)"
    )
    
    parser.add_argument(
        "--nodryrun", 
        action="store_true",
        help="禁用干运行模式，发送真实命令到机器人"
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        default=RunMode.LOCOMOTION,
        choices=[RunMode.SPORT, RunMode.STAND, RunMode.LOCOMOTION, RunMode.WALK],
        help=f"运行模式 (默认: {RunMode.LOCOMOTION})"
    )
    
    parser.add_argument(
        "--loop_mode", 
        type=str, 
        default="timer",
        choices=["timer", "while"],
        help="控制循环模式 (默认: timer)"
    )
    
    parser.add_argument(
        "--logdir", 
        type=str, 
        default=None,
        help="日志目录路径"
    )
    
    args = parser.parse_args()
    
    # 验证参数
    if args.model_dir and not os.path.exists(args.model_dir):
        print(f"错误: 模型目录不存在: {args.model_dir}")
        return 1
    
    if args.logdir and not os.path.exists(args.logdir):
        print(f"错误: 日志目录不存在: {args.logdir}")
        return 1
    
    # 设置干运行模式
    args.dryrun = not args.nodryrun
    
    try:
        # 初始化ROS2
        rclpy.init()
        
        # 创建并启动运行器
        runner = LocomotionPolicyRunner(args)
        
        if not runner.start():
            print("运行器启动失败")
            return 1
        
        # 运行ROS2循环
        rclpy.spin(runner)
        
    except KeyboardInterrupt:
        print("\n收到中断信号，正在关闭...")
    except Exception as e:
        print(f"运行时错误: {e}")
        return 1
    finally:
        # 关闭运行器
        if 'runner' in locals():
            runner.shutdown()
        
        # 关闭ROS2
        rclpy.shutdown()
    
    print("程序正常退出")
    return 0


if __name__ == "__main__":
    sys.exit(main())
