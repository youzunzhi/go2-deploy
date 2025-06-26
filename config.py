# config.py
# 配置管理

import json
import os
from collections import OrderedDict
from typing import Dict, Any, Optional
from constants import RobotConfig, ObservationConfig, ControlConfig, RunMode


class RobotConfiguration:
    """机器人配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.config_dict = {}
        self.config_path = config_path
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        else:
            self.set_default_config()
    
    def load_config(self, config_path: str):
        """加载配置文件"""
        try:
            with open(config_path, "r") as f:
                self.config_dict = json.load(f, object_pairs_hook=OrderedDict)
            print(f"配置已从 {config_path} 加载")
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            self.set_default_config()
    
    def set_default_config(self):
        """设置默认配置"""
        self.config_dict = {
            "control": {
                "control_type": "P",
                "action_scale": 1.0,
                "computer_clip_torque": True,
                "stiffness": {
                    "hip": 80.0,
                    "thigh": 80.0,
                    "calf": 80.0
                },
                "damping": {
                    "hip": 1.0,
                    "thigh": 1.0,
                    "calf": 1.0
                }
            },
            "normalization": {
                "clip_observations": 10.0,
                "clip_actions": 1.0,
                "clip_actions_method": "hard",
                "clip_actions_high": [1.0] * RobotConfig.num_actions,
                "clip_actions_low": [-1.0] * RobotConfig.num_actions,
                "obs_scales": {
                    "ang_vel": 0.25,
                    "dof_pos": 1.0,
                    "dof_vel": 0.05
                }
            },
            "init_state": {
                "default_joint_angles": {
                    "FR_hip_joint": 0.0,
                    "FR_thigh_joint": 0.8,
                    "FR_calf_joint": -1.6,
                    "FL_hip_joint": 0.0,
                    "FL_thigh_joint": 0.8,
                    "FL_calf_joint": -1.6,
                    "RR_hip_joint": 0.0,
                    "RR_thigh_joint": 0.8,
                    "RR_calf_joint": -1.6,
                    "RL_hip_joint": 0.0,
                    "RL_thigh_joint": 0.8,
                    "RL_calf_joint": -1.6
                }
            }
        }
        print("使用默认配置")
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split('.')
        value = self.config_dict
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """设置配置值"""
        keys = key.split('.')
        config = self.config_dict
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_control_config(self) -> Dict[str, Any]:
        """获取控制配置"""
        return self.config_dict.get("control", {})
    
    def get_normalization_config(self) -> Dict[str, Any]:
        """获取归一化配置"""
        return self.config_dict.get("normalization", {})
    
    def get_init_state_config(self) -> Dict[str, Any]:
        """获取初始状态配置"""
        return self.config_dict.get("init_state", {})
    
    def get_stiffness_gains(self) -> list:
        """获取刚度增益"""
        stiffness_config = self.get("control.stiffness", {})
        gains = []
        
        for joint_name in RobotConfig.dof_names:
            gain_found = False
            for key, value in stiffness_config.items():
                if key in joint_name:
                    gains.append(value)
                    gain_found = True
                    break
            if not gain_found:
                gains.append(80.0)  # 默认值
        
        return gains
    
    def get_damping_gains(self) -> list:
        """获取阻尼增益"""
        damping_config = self.get("control.damping", {})
        gains = []
        
        for joint_name in RobotConfig.dof_names:
            gain_found = False
            for key, value in damping_config.items():
                if key in joint_name:
                    gains.append(value)
                    gain_found = True
                    break
            if not gain_found:
                gains.append(1.0)  # 默认值
        
        return gains
    
    def get_default_joint_angles(self) -> Dict[str, float]:
        """获取默认关节角度"""
        return self.get("init_state.default_joint_angles", {})
    
    def save_config(self, save_path: str):
        """保存配置到文件"""
        try:
            with open(save_path, "w") as f:
                json.dump(self.config_dict, f, indent=2)
            print(f"配置已保存到 {save_path}")
        except Exception as e:
            print(f"保存配置文件失败: {e}")
    
    def print_config(self):
        """打印配置信息"""
        print("当前配置:")
        print(json.dumps(self.config_dict, indent=2, ensure_ascii=False))


class DeploymentConfig:
    """部署配置"""
    
    def __init__(self):
        self.device = "cuda"  # 推理设备
        self.duration = ControlConfig.default_duration  # 控制周期
        self.dryrun = True  # 是否干运行模式
        self.mode = RunMode.LOCOMOTION  # 运行模式
        self.loop_mode = "timer"  # 循环模式
        self.logdir = None  # 日志目录
        
        # 性能配置
        self.visual_update_interval = ControlConfig.visual_update_interval
        self.warm_up_iterations = 2
        
        # 安全配置
        self.safety_ratio = 1.1
        self.enable_safety_check = True
    
    def from_args(self, args):
        """从命令行参数加载配置"""
        if hasattr(args, 'device'):
            self.device = args.device
        if hasattr(args, 'duration'):
            self.duration = args.duration
        if hasattr(args, 'dryrun'):
            self.dryrun = args.dryrun
        if hasattr(args, 'mode'):
            self.mode = args.mode
        if hasattr(args, 'loop_mode'):
            self.loop_mode = args.loop_mode
        if hasattr(args, 'logdir'):
            self.logdir = args.logdir
    
    def print_config(self):
        """打印部署配置"""
        print("部署配置:")
        print(f"  设备: {self.device}")
        print(f"  控制周期: {self.duration}s")
        print(f"  干运行模式: {self.dryrun}")
        print(f"  运行模式: {self.mode}")
        print(f"  循环模式: {self.loop_mode}")
        print(f"  日志目录: {self.logdir}")
        print(f"  视觉更新间隔: {self.visual_update_interval}")
        print(f"  预热迭代次数: {self.warm_up_iterations}")
        print(f"  安全检查比例: {self.safety_ratio}")
        print(f"  启用安全检查: {self.enable_safety_check}") 