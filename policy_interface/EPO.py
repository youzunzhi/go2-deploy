import torch
import torch.nn as nn
import os
from collections import OrderedDict
import json
import os.path as osp

from .base import BasePolicyInterface
from utils import get_joint_map_from_names, parse_default_joint_pos_dict

class EPOPolicyInterface(BasePolicyInterface):
    def __init__(self, logdir, device):
        super().__init__(logdir, device)
        self._load_configs()
        estimator, hist_encoder = self._load_base_model()
        depth_encoder = self._load_vision_model()
        self.obs_manager = ObsManager(device, depth_encoder, estimator, hist_encoder, self.n_hist_len, self.n_proprio, self.obs_scales)

    def get_action(self):
        self.policy_iter_counter += 1
        obs = self._get_obs()
        action = self._get_action_from_obs(obs)
        return action
    
    def _get_obs(self):
        assert self.handler is not None, "Handler is not set"
        policy_iter_counter = self.policy_iter_counter
        depth_image = self.handler.get_depth_image()
        ang_vel = self.handler.get_ang_vel_obs()
        base_rpy = self.handler.get_base_rpy_obs()
        dof_pos = self.handler.get_dof_pos_obs()
        dof_vel = self.handler.get_dof_vel_obs()
        last_actions = self.handler.get_last_actions_obs()
        contact = self.handler.get_contact_filt_obs()
        return self.obs_manager.get_obs(policy_iter_counter, depth_image, ang_vel, base_rpy, dof_pos, dof_vel, last_actions, contact)
    
    def _get_action_from_obs(self, obs):
        return self.actor(obs)

    def _load_base_model(self):
        base_model_name = 'base_jit.pt'
        base_model_path = os.path.join(self.logdir, base_model_name)
        
        # Load base model in JIT format
        base_model = torch.jit.load(base_model_path, map_location=self.device)  # type: ignore
        base_model.eval()
        
        # Extract model components
        estimator = base_model.estimator.estimator
        hist_encoder = base_model.actor.history_encoder
        self.actor = base_model.actor.actor_backbone
        
        return estimator, hist_encoder

    def _load_vision_model(self):
        """
        Load vision model (depth encoder)
        
        Args:
            logdir: Model file directory
            device: Computing device
            
        Returns:
            depth_encoder: Depth encoder model
        """
        vision_model_name = 'vision_weight.pt'
        vision_model_path = os.path.join(self.logdir, vision_model_name)
        
        # Load vision model weights
        vision_model = torch.load(vision_model_path, map_location=self.device)
        
        # Create depth encoder
        depth_backbone = DepthOnlyFCBackbone58x87(None, 32, 512)
        depth_encoder = RecurrentDepthBackbone(depth_backbone, None).to(self.device)
        
        # Load pre-trained weights
        depth_encoder.load_state_dict(vision_model['depth_encoder_state_dict'])
        depth_encoder.to(self.device)
        depth_encoder.eval()
        
        return depth_encoder

    def _load_configs(self):
        config_path = osp.join(self.logdir, "config.json")
        with open(config_path, "r") as f:
            full_config = json.load(f, object_pairs_hook=OrderedDict)

        # EPO uses identity mapping (simulation order already matches hardware)
        joint_names = [
            "FR_hip_joint",     # 0
            "FR_thigh_joint",   # 1
            "FR_calf_joint",    # 2
            "FL_hip_joint",     # 3
            "FL_thigh_joint",   # 4
            "FL_calf_joint",    # 5
            "RR_hip_joint",     # 6
            "RR_thigh_joint",   # 7
            "RR_calf_joint",    # 8
            "RL_hip_joint",     # 9
            "RL_thigh_joint",   # 10
            "RL_calf_joint",    # 11
        ]
        self.joint_map = get_joint_map_from_names(joint_names)
        default_joint_pos_dict = full_config.get("init_state", {}).get("default_joint_angles", {})
        self.default_joint_pos = parse_default_joint_pos_dict(default_joint_pos_dict, joint_names)
        self.kp = full_config.get("control", {}).get("stiffness", {}).get("joint", 40.)
        self.kd = full_config.get("control", {}).get("damping", {}).get("joint", 1.)
        self.action_scale = full_config.get("control", {}).get("action_scale", 0.25)
        self.clip_obs = full_config["normalization"]["clip_obs"]
        self.clip_actions = full_config["normalization"]["clip_actions"]

        self.n_hist_len = full_config["env"]["history_len"]
        self.n_proprio = full_config["env"]["n_proprio"]
        self.obs_scales = full_config["normalization"]["obs_scales"]


class ObsManager:
    def __init__(self, device, depth_encoder, estimator, hist_encoder, n_hist_len, n_proprio, obs_scales):
        self.device = device
        self.depth_encoder = depth_encoder
        self.estimator = estimator
        self.hist_encoder = hist_encoder
        self.visual_update_interval = 5
        parkour_walk_mode = "walk"
        print(f"Which mode (parkour or walk): {parkour_walk_mode}")
        if parkour_walk_mode == "parkour":
            self.parkour_walk_tensor = torch.tensor([[1, 0]], device=self.device, dtype=torch.float32)
        elif parkour_walk_mode == "walk":
            self.parkour_walk_tensor = torch.tensor([[0, 1]], device=self.device, dtype=torch.float32)
        self.n_hist_len = n_hist_len
        self.n_proprio = n_proprio
        self.obs_scales = obs_scales

        self.last_depth_image = None
        self.proprio_history_buf = torch.zeros(1, self.n_hist_len, self.n_proprio, device=self.device, dtype=torch.float)
        self.episode_length_buf = torch.zeros(1, device=self.device, dtype=torch.float)

        self.activation = nn.ELU()

    def get_obs(self, policy_iter_counter, depth_image, ang_vel, base_rpy, dof_pos, dof_vel, last_actions, contact):
        proprio = self.get_proprio(ang_vel, base_rpy, dof_pos, dof_vel, last_actions, contact)
        if policy_iter_counter % self.visual_update_interval == 0:
            if policy_iter_counter == 0:
                last_depth_image = depth_image
            depth_latent_yaw = self.depth_encoder(last_depth_image, proprio)
        self.last_depth_image = depth_image
        # Separate depth features and yaw angle
        depth_latent = depth_latent_yaw[:, :-2]
        yaw = depth_latent_yaw[:, -2:] * 1.5
        
        # Update yaw angle in proprioceptive data
        proprio[:, 6:8] = yaw
        
        # Estimate linear velocity features
        lin_vel_latent = self.estimator(proprio)
        
        # Process historical proprioceptive data
        priv_latent = self.hist_encoder(self.activation, self.proprio_history_buf.view(-1, self.n_hist_len, self.n_proprio))
        
        # Concatenate all features
        obs = torch.cat([proprio, depth_latent, lin_vel_latent, priv_latent], dim=-1)
        
        return obs

    def get_proprio(self, ang_vel, base_rpy, dof_pos, dof_vel, last_actions, contact):
        ang_vel = ang_vel * self.obs_scales["ang_vel"]
        
        imu = base_rpy[:, 0:2]
        
        yaw_info = torch.tensor([[0, 0, 0]], device=self.device, dtype=torch.float32)

        commands = torch.tensor([[0, 0, 0.5]], device=self.device, dtype=torch.float32)
        
        dof_pos = dof_pos * self.obs_scales["dof_pos"]
        dof_vel = dof_vel * self.obs_scales["dof_vel"]
        
        proprio = torch.cat([ang_vel, imu, yaw_info, commands, self.parkour_walk_tensor,
                        dof_pos, dof_vel,
                        last_actions, 
                        contact], dim=-1)

        self.proprio_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([proprio] * self.n_hist_len, dim=1),
            torch.cat([
                self.proprio_history_buf[:, 1:],
                proprio.unsqueeze(1)
            ], dim=1)
        )

        self.episode_length_buf += 1

        return proprio


class DepthOnlyFCBackbone58x87(nn.Module):
    def __init__(self, prop_dim, scandots_output_dim, hidden_state_dim, output_activation=None, num_frames=1):
        super().__init__()

        self.num_frames = num_frames
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            # [1, 58, 87]
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            # [32, 54, 83]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 27, 41]
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            activation,
            nn.Flatten(),
            # [32, 25, 39]
            nn.Linear(64 * 25 * 39, 128),
            activation,
            nn.Linear(128, scandots_output_dim)
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

    def forward(self, images: torch.Tensor):
        images_compressed = self.image_compression(images.unsqueeze(1))
        latent = self.output_activation(images_compressed)

        return latent


class RecurrentDepthBackbone(nn.Module):
    def __init__(self, base_backbone, env_cfg) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = base_backbone
        if env_cfg == None:
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
        self.rnn = nn.GRU(input_size=32, hidden_size=512, batch_first=True)
        # 为什么要 32 + 2, 根据之后的代码，scandot latent 是 32 维的
        # 但 backbone 代码明确写了，和 scandot latent 维度是一样的
        # 所以这个有 recurrent 的 depth backbone 另有他用
        self.output_mlp = nn.Sequential(
                                nn.Linear(512, 32+2),
                                last_activation
                            )
        self.hidden_states = None

    def forward(self, depth_image, proprioception):
        # 先处理深度图像，得到一个基础的特征表示，32维的 latent 数据
        depth_image = self.base_backbone(depth_image)
        # 再把 proprioception 和基础的图像 latent 拼接在一起，传入一个MLP处理 
        depth_latent = self.combination_mlp(torch.cat((depth_image, proprioception), dim=-1))
        # 最后通过 RNN 得到最终的 latent 表示
        depth_latent, self.hidden_states = self.rnn(depth_latent[:, None, :], self.hidden_states)
        # 根据 output 全连接网络来看，输出维度应该是 34？
        depth_latent = self.output_mlp(depth_latent.squeeze(1))
        
        return depth_latent

    def detach_hidden_states(self):
        self.hidden_states = self.hidden_states.detach().clone()

