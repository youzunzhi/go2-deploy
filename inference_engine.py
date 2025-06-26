 # inference_engine.py
# 推理引擎

import torch
import torch.nn as nn
import time
from typing import Optional, Dict, Any, Tuple
import numpy as np

from constants import RobotConfig, ObservationConfig, ControlConfig
from model_manager import ModelManager
from utils import PerformanceTimer


class InferenceEngine:
    """推理引擎"""
    
    def __init__(self, 
                 model_dir: str, 
                 device: str = "cuda",
                 warm_up_iterations: int = 2):
        """
        初始化推理引擎
        
        Args:
            model_dir: 模型目录路径
            device: 推理设备
            warm_up_iterations: 预热迭代次数
        """
        self.device = device
        self.warm_up_iterations = warm_up_iterations
        
        # 模型管理器
        self.model_manager = ModelManager(model_dir, device)
        
        # 性能监控
        self.performance_timer = PerformanceTimer()
        self.inference_count = 0
        self.total_inference_time = 0.0
        
        # 状态管理
        self.is_initialized = False
        self.last_depth_image = None
        self.visual_update_interval = ControlConfig.visual_update_interval
        
        # 缓冲区
        self._init_buffers()
        
        print(f"推理引擎初始化完成，设备: {device}")
    
    def _init_buffers(self):
        """初始化缓冲区"""
        # 观察缓冲区
        self.proprio_history_buf = torch.zeros(
            1, ObservationConfig.n_hist_len, ObservationConfig.n_proprio,
            device=self.device, dtype=torch.float
        )
        self.episode_length_buf = torch.zeros(
            1, device=self.device, dtype=torch.float
        )
        self.depth_latent_yaw_buffer = torch.zeros(
            1, ObservationConfig.n_depth_latent + 2,
            device=self.device, dtype=torch.float
        )
        
        # 性能统计
        self.inference_times = []
        self.max_inference_times = 100  # 保留最近100次的推理时间
    
    def initialize(self) -> bool:
        """
        初始化推理引擎
        
        Returns:
            bool: 是否成功初始化
        """
        try:
            print("开始初始化推理引擎...")
            
            # 加载模型
            if not self.model_manager.load_models():
                print("模型加载失败")
                return False
            
            # 预热模型
            self.model_manager.warm_up(self.warm_up_iterations)
            
            # 重置状态
            self.reset()
            
            self.is_initialized = True
            print("推理引擎初始化完成")
            
            return True
            
        except Exception as e:
            print(f"推理引擎初始化失败: {e}")
            return False
    
    def reset(self):
        """重置推理引擎状态"""
        # 重置缓冲区
        self.proprio_history_buf = torch.zeros(
            1, ObservationConfig.n_hist_len, ObservationConfig.n_proprio,
            device=self.device, dtype=torch.float
        )
        self.episode_length_buf = torch.zeros(
            1, device=self.device, dtype=torch.float
        )
        self.depth_latent_yaw_buffer = torch.zeros(
            1, ObservationConfig.n_depth_latent + 2,
            device=self.device, dtype=torch.float
        )
        
        # 重置深度编码器
        self.model_manager.reset_depth_encoder()
        
        # 重置性能统计
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.inference_times = []
        
        # 重置深度图像
        self.last_depth_image = None
        
        print("推理引擎状态已重置")
    
    def update_proprio_history(self, proprio: torch.Tensor):
        """
        更新本体感受历史
        
        Args:
            proprio: 当前本体感受 [batch_size, n_proprio]
        """
        self.proprio_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([proprio] * ObservationConfig.n_hist_len, dim=1),
            torch.cat([
                self.proprio_history_buf[:, 1:],
                proprio.unsqueeze(1)
            ], dim=1)
        )
        self.episode_length_buf += 1
    
    def should_update_vision(self, counter: int) -> bool:
        """
        判断是否应该更新视觉
        
        Args:
            counter: 当前计数器
            
        Returns:
            bool: 是否应该更新视觉
        """
        return counter % self.visual_update_interval == 0
    
    def process_depth_image(self, depth_image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """
        处理深度图像
        
        Args:
            depth_image: 深度图像 [batch_size, 58, 87]
            proprio: 本体感受 [batch_size, n_proprio]
            
        Returns:
            torch.Tensor: 深度特征 [batch_size, 34]
        """
        if depth_image is None:
            return torch.zeros(
                proprio.shape[0], ObservationConfig.n_depth_latent + 2,
                device=self.device, dtype=torch.float32
            )
        
        # 编码深度图像
        depth_latent_yaw = self.model_manager.encode_depth(depth_image, proprio)
        
        return depth_latent_yaw
    
    def inference_step(self, 
                      proprio: torch.Tensor,
                      depth_image: Optional[torch.Tensor] = None,
                      counter: int = 0) -> torch.Tensor:
        """
        推理步骤
        
        Args:
            proprio: 本体感受 [batch_size, n_proprio]
            depth_image: 深度图像 [batch_size, 58, 87] (可选)
            counter: 当前计数器
            
        Returns:
            torch.Tensor: 动作 [batch_size, num_actions]
        """
        if not self.is_initialized:
            raise RuntimeError("推理引擎未初始化")
        
        start_time = time.monotonic()
        
        # 更新本体感受历史
        self.update_proprio_history(proprio)
        
        # 处理深度图像
        if self.should_update_vision(counter) and depth_image is not None:
            self.depth_latent_yaw_buffer = self.process_depth_image(depth_image, proprio)
            self.last_depth_image = depth_image
        elif self.last_depth_image is not None:
            # 使用上一帧的深度图像
            self.depth_latent_yaw_buffer = self.process_depth_image(self.last_depth_image, proprio)
        
        # 完整推理
        action = self.model_manager.inference_step(
            proprio, 
            depth_image if self.should_update_vision(counter) else None,
            self.proprio_history_buf
        )
        
        # 记录性能
        inference_time = time.monotonic() - start_time
        self._update_performance_stats(inference_time)
        
        return action
    
    def _update_performance_stats(self, inference_time: float):
        """更新性能统计"""
        self.inference_count += 1
        self.total_inference_time += inference_time
        
        # 记录推理时间
        self.inference_times.append(inference_time)
        if len(self.inference_times) > self.max_inference_times:
            self.inference_times.pop(0)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        if self.inference_count == 0:
            return {
                "inference_count": 0,
                "avg_inference_time": 0.0,
                "min_inference_time": 0.0,
                "max_inference_time": 0.0,
                "total_inference_time": 0.0
            }
        
        return {
            "inference_count": self.inference_count,
            "avg_inference_time": self.total_inference_time / self.inference_count,
            "min_inference_time": min(self.inference_times),
            "max_inference_time": max(self.inference_times),
            "total_inference_time": self.total_inference_time,
            "recent_avg_time": np.mean(self.inference_times),
            "recent_std_time": np.std(self.inference_times)
        }
    
    def print_performance_stats(self):
        """打印性能统计"""
        stats = self.get_performance_stats()
        
        print("推理性能统计:")
        print(f"  推理次数: {stats['inference_count']}")
        print(f"  平均推理时间: {stats['avg_inference_time']:.5f}s")
        print(f"  最小推理时间: {stats['min_inference_time']:.5f}s")
        print(f"  最大推理时间: {stats['max_inference_time']:.5f}s")
        print(f"  最近平均时间: {stats['recent_avg_time']:.5f}s")
        print(f"  最近标准差: {stats['recent_std_time']:.5f}s")
        
        # 打印模型管理器性能
        self.model_manager.print_performance_stats()
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return self.model_manager.get_model_info()
    
    def cleanup(self):
        """清理资源"""
        if self.model_manager:
            self.model_manager.cleanup()
        
        # 清理GPU内存
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        print("推理引擎资源已清理")


class InferencePipeline:
    """推理流水线"""
    
    def __init__(self, inference_engine: InferenceEngine):
        """
        初始化推理流水线
        
        Args:
            inference_engine: 推理引擎
        """
        self.inference_engine = inference_engine
        self.performance_timer = PerformanceTimer()
        
        # 流水线状态
        self.is_running = False
        self.counter = 0
        
        print("推理流水线初始化完成")
    
    def start(self):
        """启动流水线"""
        if not self.inference_engine.is_initialized:
            raise RuntimeError("推理引擎未初始化")
        
        self.is_running = True
        self.counter = 0
        print("推理流水线已启动")
    
    def stop(self):
        """停止流水线"""
        self.is_running = False
        print("推理流水线已停止")
    
    def step(self, 
             proprio: torch.Tensor,
             depth_image: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        执行一步推理
        
        Args:
            proprio: 本体感受
            depth_image: 深度图像（可选）
            
        Returns:
            torch.Tensor: 动作
        """
        if not self.is_running:
            raise RuntimeError("推理流水线未运行")
        
        self.performance_timer.start()
        
        # 执行推理
        action = self.inference_engine.inference_step(
            proprio, depth_image, self.counter
        )
        
        self.counter += 1
        self.performance_timer.checkpoint("total_step")
        
        return action
    
    def reset(self):
        """重置流水线"""
        self.inference_engine.reset()
        self.counter = 0
        print("推理流水线已重置")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        engine_stats = self.inference_engine.get_performance_stats()
        
        return {
            "pipeline_counter": self.counter,
            "is_running": self.is_running,
            "engine_stats": engine_stats
        }
    
    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        
        print("推理流水线统计:")
        print(f"  计数器: {stats['pipeline_counter']}")
        print(f"  运行状态: {stats['is_running']}")
        
        self.inference_engine.print_performance_stats()