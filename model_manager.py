 # model_manager.py
# 模型管理器

import os
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
import json
from collections import OrderedDict

from constants import RobotConfig, ObservationConfig, ControlConfig
from utils import PerformanceTimer


class DepthOnlyFCBackbone58x87(nn.Module):
    """深度图像编码器 - 58x87分辨率"""
    
    def __init__(self, prop_dim, scandots_output_dim, hidden_state_dim, output_activation=None, num_frames=1):
        super().__init__()
        
        self.num_frames = num_frames
        activation = nn.ELU()
        
        # 图像压缩网络
        self.image_compression = nn.Sequential(
            # 输入: [1, 58, 87]
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            # 输出: [32, 54, 83]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 输出: [32, 27, 41]
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            activation,
            nn.Flatten(),
            # 输出: [64, 25, 39]
            nn.Linear(64 * 25 * 39, 128),
            activation,
            nn.Linear(128, scandots_output_dim)
        )
        
        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation
    
    def forward(self, images: torch.Tensor):
        """前向传播"""
        images_compressed = self.image_compression(images.unsqueeze(1))
        latent = self.output_activation(images_compressed)
        return latent


class RecurrentDepthBackbone(nn.Module):
    """循环深度编码器"""
    
    def __init__(self, base_backbone, env_cfg) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        
        self.base_backbone = base_backbone
        
        # 组合MLP: 深度特征 + 本体感受 -> 32维
        if env_cfg is None:
            self.combination_mlp = nn.Sequential(
                nn.Linear(32 + 53, 128),
                activation,
                nn.Linear(128, 32)
            )
        else:
            self.combination_mlp = nn.Sequential(
                nn.Linear(32 + env_cfg.env.n_proprio, 128),
                activation,
                nn.Linear(128, 32)
            )
        
        # RNN层
        self.rnn = nn.GRU(input_size=32, hidden_size=512, batch_first=True)
        
        # 输出MLP: 512 -> 34 (32 + 2)
        self.output_mlp = nn.Sequential(
            nn.Linear(512, 32 + 2),
            last_activation
        )
        
        self.hidden_states = None
    
    def forward(self, depth_image, proprioception):
        """前向传播"""
        # 基础深度特征提取
        depth_image = self.base_backbone(depth_image)
        
        # 组合深度特征和本体感受
        depth_latent = self.combination_mlp(
            torch.cat((depth_image, proprioception), dim=-1)
        )
        
        # RNN处理
        depth_latent, self.hidden_states = self.rnn(
            depth_latent[:, None, :], self.hidden_states
        )
        
        # 输出处理
        depth_latent = self.output_mlp(depth_latent.squeeze(1))
        
        return depth_latent
    
    def detach_hidden_states(self):
        """分离隐藏状态"""
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.detach().clone()


class ModelManager:
    """模型管理器"""
    
    def __init__(self, model_dir: str, device: str = "cuda"):
        """
        初始化模型管理器
        
        Args:
            model_dir: 模型目录路径
            device: 推理设备
        """
        self.model_dir = model_dir
        self.device = device
        self.performance_timer = PerformanceTimer()
        
        # 模型组件
        self.base_model = None
        self.depth_encoder = None
        self.estimator = None
        self.hist_encoder = None
        self.actor = None
        
        # 模型配置
        self.model_config = None
        
        # 状态
        self.is_loaded = False
        self.is_warmed_up = False
        
        print(f"模型管理器初始化完成，设备: {device}")
    
    def load_models(self) -> bool:
        """
        加载模型
        
        Returns:
            bool: 是否成功加载
        """
        try:
            self.performance_timer.start()
            
            # 加载配置文件
            config_path = os.path.join(self.model_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    self.model_config = json.load(f, object_pairs_hook=OrderedDict)
                print(f"加载模型配置: {config_path}")
            
            # 加载基础模型
            base_model_path = os.path.join(self.model_dir, "base_jit.pt")
            if not os.path.exists(base_model_path):
                raise FileNotFoundError(f"基础模型文件不存在: {base_model_path}")
            
            self.base_model = torch.jit.load(base_model_path, map_location=self.device)
            self.base_model.eval()
            self.performance_timer.checkpoint("base_model")
            
            # 提取模型组件
            self._extract_model_components()
            self.performance_timer.checkpoint("extract_components")
            
            # 加载视觉模型
            vision_model_path = os.path.join(self.model_dir, "vision_weight.pt")
            if os.path.exists(vision_model_path):
                self._load_vision_model(vision_model_path)
                self.performance_timer.checkpoint("vision_model")
            else:
                print("警告: 视觉模型文件不存在，将使用默认配置")
                self._create_default_vision_model()
            
            self.is_loaded = True
            self.performance_timer.checkpoint("total_load")
            
            print("模型加载完成")
            self.performance_timer.print_timings("模型加载: ")
            
            return True
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False
    
    def _extract_model_components(self):
        """提取模型组件"""
        try:
            # 提取估计器
            self.estimator = self.base_model.estimator.estimator
            self.estimator.eval()
            
            # 提取历史编码器
            self.hist_encoder = self.base_model.actor.history_encoder
            self.hist_encoder.eval()
            
            # 提取动作网络
            self.actor = self.base_model.actor.actor_backbone
            self.actor.eval()
            
            print("模型组件提取完成")
            
        except Exception as e:
            print(f"模型组件提取失败: {e}")
            raise
    
    def _load_vision_model(self, vision_model_path: str):
        """加载视觉模型"""
        try:
            vision_model = torch.load(vision_model_path, map_location=self.device)
            
            # 创建深度编码器
            depth_backbone = DepthOnlyFCBackbone58x87(None, 32, 512)
            self.depth_encoder = RecurrentDepthBackbone(depth_backbone, None).to(self.device)
            
            # 加载权重
            self.depth_encoder.load_state_dict(vision_model['depth_encoder_state_dict'])
            self.depth_encoder.eval()
            
            print("视觉模型加载完成")
            
        except Exception as e:
            print(f"视觉模型加载失败: {e}")
            raise
    
    def _create_default_vision_model(self):
        """创建默认视觉模型"""
        try:
            depth_backbone = DepthOnlyFCBackbone58x87(None, 32, 512)
            self.depth_encoder = RecurrentDepthBackbone(depth_backbone, None).to(self.device)
            self.depth_encoder.eval()
            
            print("创建默认视觉模型")
            
        except Exception as e:
            print(f"创建默认视觉模型失败: {e}")
            raise
    
    def warm_up(self, num_iterations: int = 2):
        """
        预热模型
        
        Args:
            num_iterations: 预热迭代次数
        """
        if not self.is_loaded:
            print("错误: 模型未加载，无法预热")
            return
        
        print(f"开始模型预热，迭代次数: {num_iterations}")
        
        try:
            self.performance_timer.start()
            
            # 创建测试数据
            test_proprio = torch.ones(1, ObservationConfig.n_proprio, device=self.device)
            test_depth = torch.ones(1, 58, 87, device=self.device)
            test_proprio_history = torch.ones(
                1, ObservationConfig.n_hist_len, ObservationConfig.n_proprio, 
                device=self.device
            )
            
            # 预热迭代
            for i in range(num_iterations):
                with torch.no_grad():
                    # 预热深度编码器
                    if self.depth_encoder is not None:
                        depth_latent = self.depth_encoder(test_depth, test_proprio)
                    
                    # 预热估计器
                    lin_vel_latent = self.estimator(test_proprio)
                    
                    # 预热历史编码器
                    activation = nn.ELU()
                    priv_latent = self.hist_encoder(
                        activation, 
                        test_proprio_history.view(-1, ObservationConfig.n_hist_len, ObservationConfig.n_proprio)
                    )
                    
                    # 预热动作网络
                    if self.depth_encoder is not None:
                        depth_latent_features = depth_latent[:, :-2]
                        obs = torch.cat([test_proprio, depth_latent_features, lin_vel_latent, priv_latent], dim=-1)
                    else:
                        obs = torch.cat([test_proprio, lin_vel_latent, priv_latent], dim=-1)
                    
                    action = self.actor(obs)
                
                print(f"预热迭代 {i+1}/{num_iterations} 完成")
            
            self.is_warmed_up = True
            self.performance_timer.checkpoint("total_warmup")
            
            print("模型预热完成")
            self.performance_timer.print_timings("模型预热: ")
            
        except Exception as e:
            print(f"模型预热失败: {e}")
    
    def encode_depth(self, depth_image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """
        编码深度图像
        
        Args:
            depth_image: 深度图像 [batch_size, 58, 87]
            proprio: 本体感受 [batch_size, n_proprio]
            
        Returns:
            torch.Tensor: 深度特征 [batch_size, 34]
        """
        if not self.is_loaded or self.depth_encoder is None:
            raise RuntimeError("深度编码器未加载")
        
        self.performance_timer.start()
        
        with torch.no_grad():
            depth_latent_yaw = self.depth_encoder(depth_image, proprio)
            
            # 检查NaN值
            if torch.isnan(depth_latent_yaw).any():
                print("警告: 深度编码输出包含NaN值")
                print(f"深度图像范围: [{depth_image.min():.3f}, {depth_image.max():.3f}]")
                print(f"本体感受范围: [{proprio.min():.3f}, {proprio.max():.3f}]")
        
        self.performance_timer.checkpoint("depth_encode")
        
        return depth_latent_yaw
    
    def estimate_velocity(self, proprio: torch.Tensor) -> torch.Tensor:
        """
        估计线速度
        
        Args:
            proprio: 本体感受 [batch_size, n_proprio]
            
        Returns:
            torch.Tensor: 估计的线速度 [batch_size, 9]
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载")
        
        self.performance_timer.start()
        
        with torch.no_grad():
            lin_vel_latent = self.estimator(proprio)
        
        self.performance_timer.checkpoint("velocity_estimate")
        
        return lin_vel_latent
    
    def encode_history(self, proprio_history: torch.Tensor) -> torch.Tensor:
        """
        编码历史本体感受
        
        Args:
            proprio_history: 历史本体感受 [batch_size, n_hist_len, n_proprio]
            
        Returns:
            torch.Tensor: 历史特征 [batch_size, 20]
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载")
        
        self.performance_timer.start()
        
        with torch.no_grad():
            activation = nn.ELU()
            priv_latent = self.hist_encoder(activation, proprio_history)
        
        self.performance_timer.checkpoint("history_encode")
        
        return priv_latent
    
    def predict_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        预测动作
        
        Args:
            obs: 观察向量 [batch_size, obs_dim]
            
        Returns:
            torch.Tensor: 动作 [batch_size, num_actions]
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载")
        
        self.performance_timer.start()
        
        with torch.no_grad():
            action = self.actor(obs)
        
        self.performance_timer.checkpoint("action_predict")
        
        return action
    
    def process_observation(self, 
                          proprio: torch.Tensor, 
                          depth_latent_yaw: torch.Tensor, 
                          proprio_history: torch.Tensor) -> torch.Tensor:
        """
        处理观察数据
        
        Args:
            proprio: 本体感受 [batch_size, n_proprio]
            depth_latent_yaw: 深度特征 [batch_size, 34]
            proprio_history: 历史本体感受 [batch_size, n_hist_len, n_proprio]
            
        Returns:
            torch.Tensor: 完整的观察向量 [batch_size, obs_dim]
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载")
        
        self.performance_timer.start()
        
        # 提取深度特征和偏航角
        depth_latent = depth_latent_yaw[:, :-2]  # 前32维是深度特征
        yaw = depth_latent_yaw[:, -2:] * 1.5     # 后2维是偏航角
        
        # 更新本体感受中的偏航角
        proprio_updated = proprio.clone()
        proprio_updated[:, 6:8] = yaw
        
        # 估计线速度
        lin_vel_latent = self.estimate_velocity(proprio_updated)
        
        # 编码历史
        priv_latent = self.encode_history(proprio_history)
        
        # 组合观察向量
        obs = torch.cat([proprio_updated, depth_latent, lin_vel_latent, priv_latent], dim=-1)
        
        self.performance_timer.checkpoint("total_obs_process")
        
        return obs
    
    def inference_step(self, 
                      proprio: torch.Tensor, 
                      depth_image: Optional[torch.Tensor] = None,
                      proprio_history: torch.Tensor = None) -> torch.Tensor:
        """
        完整的推理步骤
        
        Args:
            proprio: 本体感受
            depth_image: 深度图像（可选）
            proprio_history: 历史本体感受
            
        Returns:
            torch.Tensor: 动作
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载")
        
        self.performance_timer.start()
        
        # 处理深度图像
        if depth_image is not None and self.depth_encoder is not None:
            depth_latent_yaw = self.encode_depth(depth_image, proprio)
        else:
            # 如果没有深度图像，使用零向量
            depth_latent_yaw = torch.zeros(
                proprio.shape[0], 34, device=self.device, dtype=torch.float32
            )
        
        # 处理观察
        obs = self.process_observation(proprio, depth_latent_yaw, proprio_history)
        
        # 预测动作
        action = self.predict_action(obs)
        
        self.performance_timer.checkpoint("total_inference")
        
        return action
    
    def reset_depth_encoder(self):
        """重置深度编码器的隐藏状态"""
        if self.depth_encoder is not None:
            self.depth_encoder.detach_hidden_states()
            print("深度编码器隐藏状态已重置")
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "is_loaded": self.is_loaded,
            "is_warmed_up": self.is_warmed_up,
            "device": self.device,
            "model_dir": self.model_dir,
            "has_depth_encoder": self.depth_encoder is not None,
            "has_estimator": self.estimator is not None,
            "has_hist_encoder": self.hist_encoder is not None,
            "has_actor": self.actor is not None
        }
    
    def print_performance_stats(self):
        """打印性能统计"""
        self.performance_timer.print_timings("推理性能: ")
    
    def cleanup(self):
        """清理资源"""
        if self.depth_encoder is not None:
            self.depth_encoder.detach_hidden_states()
        
        # 清理GPU内存
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        print("模型管理器资源已清理")