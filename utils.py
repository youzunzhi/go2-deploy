# utils.py
# 工具函数和数学计算

import torch
import numpy as np


@torch.jit.script
def copysign(a, b):
    """复制符号函数"""
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)


@torch.jit.script
def get_euler_xyz(q):
    """从四元数计算欧拉角 (roll, pitch, yaw)"""
    qx, qy, qz, qw = 0, 1, 2, 3
    
    # roll (x轴旋转)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = 1.0 - 2.0 * (q[:, qx] * q[:, qx] + q[:, qy] * q[:, qy])
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y轴旋转)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z轴旋转)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = 1.0 - 2.0 * (q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz])
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def get_crc(data):
    """计算CRC校验码 (需要根据具体的CRC模块实现)"""
    # 这里需要根据实际的CRC模块实现
    # 暂时返回0，后续需要集成真实的CRC计算
    return 0


class PerformanceTimer:
    """性能计时器"""
    
    def __init__(self):
        self.start_time = None
        self.checkpoints = {}
    
    def start(self):
        """开始计时"""
        import time
        self.start_time = time.monotonic()
        self.checkpoints = {}
    
    def checkpoint(self, name):
        """记录检查点"""
        import time
        if self.start_time is not None:
            self.checkpoints[name] = time.monotonic() - self.start_time
    
    def get_timings(self):
        """获取所有时间统计"""
        return self.checkpoints
    
    def print_timings(self, prefix=""):
        """打印时间统计"""
        if not self.checkpoints:
            return
        
        print(f"{prefix}Timing statistics:")
        prev_time = 0
        for name, time in self.checkpoints.items():
            duration = time - prev_time
            print(f"  {name}: {duration:.5f}s")
            prev_time = time


class StateMachine:
    """简单的状态机"""
    
    def __init__(self, initial_state):
        self.current_state = initial_state
        self.state_history = [initial_state]
    
    def transition_to(self, new_state):
        """状态转换"""
        if new_state != self.current_state:
            self.current_state = new_state
            self.state_history.append(new_state)
            return True
        return False
    
    def get_current_state(self):
        """获取当前状态"""
        return self.current_state
    
    def get_state_history(self):
        """获取状态历史"""
        return self.state_history.copy()


class SafetyChecker:
    """安全检查器"""
    
    def __init__(self, joint_limits_low, joint_limits_high, safety_ratio=1.1):
        """
        初始化安全检查器
        
        Args:
            joint_limits_low: 关节下限
            joint_limits_high: 关节上限
            safety_ratio: 安全比例
        """
        self.joint_limits_low = torch.tensor(joint_limits_low)
        self.joint_limits_high = torch.tensor(joint_limits_high)
        
        # 计算安全限位
        joint_pos_mid = (self.joint_limits_high + self.joint_limits_low) / 2
        joint_pos_range = (self.joint_limits_high - self.joint_limits_low) / 2
        self.safety_high = joint_pos_mid + joint_pos_range * safety_ratio
        self.safety_low = joint_pos_mid - joint_pos_range * safety_ratio
    
    def check_joint_positions(self, joint_positions):
        """
        检查关节位置是否在安全范围内
        
        Args:
            joint_positions: 关节位置张量
            
        Returns:
            bool: 是否安全
        """
        if torch.any(joint_positions < self.safety_low) or torch.any(joint_positions > self.safety_high):
            return False
        return True
    
    def get_violation_info(self, joint_positions):
        """
        获取违规信息
        
        Args:
            joint_positions: 关节位置张量
            
        Returns:
            dict: 违规信息
        """
        violations = {
            'low_violations': joint_positions < self.safety_low,
            'high_violations': joint_positions > self.safety_high,
            'unsafe_joints': torch.where(joint_positions < self.safety_low)[0].tolist() + 
                           torch.where(joint_positions > self.safety_high)[0].tolist()
        }
        return violations 