import torch
import os
import os.path as osp
import yaml
from .base import BasePolicyInterface
from utils import get_joint_map_from_names, parse_default_joint_pos_dict


class LeggedLocoPolicyInterface(BasePolicyInterface):
    def __init__(self, logdir, device):
        super().__init__(logdir, device)
        self._load_configs()
        self._detect_policy_type()
        self._load_model()
        
        # Initialize history buffer for proprioception observations
        self.history_length = 9  # from agent.yaml
        self.proprio_obs_dim = 45  # base_ang_vel(3) + base_rpy(3) + commands(3) + joint_pos(12) + joint_vel(12) + last_actions(12)
        self.proprio_obs_buf = torch.zeros(1, self.history_length, self.proprio_obs_dim, 
                                         dtype=torch.float, device=self.device)
        
        # Initialize heightmap buffer for vision policies
        if self.is_vision_policy:
            # Heightmap dimensions from legged-loco vision task
            # voxel_size_xy = 0.06m, range_x = [-0.8, 0.2] = 1.0m, range_y = [-0.8, 0.8] = 1.6m
            # x_bins = 1.0 / 0.06 ≈ 17, y_bins = 1.6 / 0.06 ≈ 27
            self.heightmap_dims = (17, 27)  # x_bins × y_bins
            self.heightmap_size = self.heightmap_dims[0] * self.heightmap_dims[1]  # 459
            
        self.warm_up_iter = 10

    def _detect_policy_type(self):
        """Detect if this is a vision or base policy based on configuration"""
        
        # Check if lidar_sensor exists in scene configuration
        self.is_vision_policy = "lidar_sensor" in self.full_config["scene"]
        
        # Check if height_map_lidar observation exists in policy observations
        if self.is_vision_policy:
            assert self.full_config['observations']['policy']['height_scan']['func'] == "omni.isaac.leggedloco.leggedloco.mdp.observations:height_map_lidar", "Height map lidar observation not found"
        
        print(f"Detected policy type: {'Vision' if self.is_vision_policy else 'Base'}")

    def get_action(self):
        self.policy_iter_counter += 1
        obs = self._get_obs()
        action = self.policy(obs)
        return action

    def _get_obs(self):
        assert self.handler is not None, "Handler is not set"
        
        if self.is_vision_policy:
            return self._get_vision_obs()
        else:
            return self._get_base_obs()

    def _get_base_obs(self):
        """Get base policy observations (proprioceptive only)"""
        ang_vel = self.handler.get_ang_vel_obs()
        base_rpy = self.handler.get_base_rpy_obs()
        dof_pos = self.handler.get_dof_pos_obs()
        dof_vel = self.handler.get_dof_vel_obs()
        last_actions = self.handler.get_last_actions_obs()
        commands = self.handler.get_xyyaw_command()
        
        # Current proprioception observation (45 dims)
        current_obs = torch.cat([ang_vel, base_rpy, commands, dof_pos, dof_vel, last_actions], dim=-1)
        
        # Update history buffer (roll and append current)
        self.proprio_obs_buf = torch.cat([
            self.proprio_obs_buf[:, 1:],  # Remove oldest
            current_obs.unsqueeze(1)      # Add current as newest
        ], dim=1)
        
        # Flatten history buffer and concatenate with current obs
        history_obs = self.proprio_obs_buf.view(1, -1)  # Shape: (1, 9*45)
        full_obs = torch.cat([current_obs, history_obs], dim=-1)  # Shape: (1, 45+405=450)
        
        return full_obs

    def _get_vision_obs(self):
        """Get vision policy observations (proprioceptive + heightmap)"""
        # Get proprioceptive observations (same as base policy)
        ang_vel = self.handler.get_ang_vel_obs()
        base_rpy = self.handler.get_base_rpy_obs()
        dof_pos = self.handler.get_dof_pos_obs()
        dof_vel = self.handler.get_dof_vel_obs()
        last_actions = self.handler.get_last_actions_obs()
        commands = self.handler.get_xyyaw_command()
        
        # Current proprioception observation (45 dims)
        current_obs = torch.cat([ang_vel, base_rpy, commands, dof_pos, dof_vel, last_actions], dim=-1)
        
        # Get heightmap from handler (459 dims)
        heightmap = self.handler.get_heightmap_obs()
        
        # Vision policy observation structure (without history as per legged-loco)
        # Policy obs: base_ang_vel(3) + base_rpy(3) + velocity_commands(3) + 
        #            joint_pos(12) + joint_vel(12) + actions(12) + height_scan(459) = 504 dims
        full_obs = torch.cat([current_obs, heightmap], dim=-1)  # Shape: (1, 45+459=504)
        
        return full_obs
        
    def _load_model(self):
        model_path = os.path.join(self.logdir, "policy.jit")
        # Load the original JIT model - it actually works fine with correct input size!
        self.policy = torch.jit.load(model_path, map_location=self.device)
        self.policy.eval()

    def _load_configs(self):
        config_path = osp.join(self.logdir, "params/env.yaml")
        with open(config_path, "r") as f:
            self.full_config = yaml.load(f, Loader=yaml.Loader)

        # legged-loco uses grouped joint order: [all hips, all thighs, all calves]
        # isaaclab sim: FL_hip(0), FR_hip(1), RL_hip(2), RR_hip(3), FL_thigh(4), FR_thigh(5), RL_thigh(6), RR_thigh(7), FL_calf(8), FR_calf(9), RL_calf(10), RR_calf(11)
        joint_names = [
            "FL_hip_joint",     # 0
            "FR_hip_joint",     # 1  
            "RL_hip_joint",     # 2
            "RR_hip_joint",     # 3
            "FL_thigh_joint",   # 4
            "FR_thigh_joint",   # 5
            "RL_thigh_joint",   # 6
            "RR_thigh_joint",   # 7
            "FL_calf_joint",    # 8
            "FR_calf_joint",    # 9
            "RL_calf_joint",    # 10
            "RR_calf_joint",    # 11
        ]
        # joint_map = [
        #     3, 0, 9, 6,   # Hip joints: FL->3, FR->0, RL->9, RR->6  
        #     4, 1, 10, 7,  # Thigh joints: FL->4, FR->1, RL->10, RR->7
        #     5, 2, 11, 8   # Calf joints: FL->5, FR->2, RL->11, RR->8
        # ]
        self.joint_map = get_joint_map_from_names(joint_names)
        default_joint_pos_dict = self.full_config.get("scene", {}).get("robot", {}).get("init_state", {}).get("joint_pos", {})
        self.default_joint_pos = parse_default_joint_pos_dict(default_joint_pos_dict, joint_names)
        self.kp = self.full_config.get("scene", {}).get("robot", {}).get("actuators", {}).get("base_legs", {}).get("stiffness", 40.0)
        self.kd = self.full_config.get("scene", {}).get("robot", {}).get("actuators", {}).get("base_legs", {}).get("damping", 1.0)
        self.action_scale = self.full_config.get("actions", {}).get("joint_pos", {}).get("scale", 0.25)
        self.clip_obs = 100.0
        self.clip_actions = None
