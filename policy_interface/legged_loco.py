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
        self._load_model()
        
        # Initialize history buffer for proprioception observations
        self.history_length = 9  # from agent.yaml
        self.proprio_obs_dim = 45  # base_ang_vel(3) + base_rpy(3) + commands(3) + joint_pos(12) + joint_vel(12) + last_actions(12)
        self.proprio_obs_buf = torch.zeros(1, self.history_length, self.proprio_obs_dim, 
                                         dtype=torch.float, device=self.device)
        
    def get_action(self):
        self.policy_iter_counter += 1
        obs = self._get_obs()
        action = self.policy(obs)
        return action

    def _get_obs(self):
        assert self.handler is not None, "Handler is not set"
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
        
    def _load_model(self):
        model_path = os.path.join(self.logdir, "policy.jit")
        # Load the original JIT model - it actually works fine with correct input size!
        self.policy = torch.jit.load(model_path, map_location=self.device)
        self.policy.eval()

    def _load_configs(self):
        config_path = osp.join(self.logdir, "params/env.yaml")
        with open(config_path, "r") as f:
            full_config = yaml.load(f, Loader=yaml.Loader)

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
        default_joint_pos_dict = full_config.get("scene", {}).get("robot", {}).get("init_state", {}).get("joint_pos", {})
        self.default_joint_pos = parse_default_joint_pos_dict(default_joint_pos_dict, joint_names)
        self.kp = full_config.get("scene", {}).get("robot", {}).get("actuators", {}).get("base_legs", {}).get("stiffness", 40.0)
        self.kd = full_config.get("scene", {}).get("robot", {}).get("actuators", {}).get("base_legs", {}).get("damping", 1.0)
        self.action_scale = full_config.get("actions", {}).get("joint_pos", {}).get("scale", 0.25)
        self.clip_obs = 100.0
        self.clip_actions = None
