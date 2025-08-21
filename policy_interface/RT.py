import torch
import os
import os.path as osp
import yaml
import numpy as np
from .base import BasePolicyInterface
from utils import get_joint_map_from_names, parse_default_joint_pos_dict
from utils.quaternion_utils import quat_rotate_inverse


class RapidTurnPolicyInterface(BasePolicyInterface):
    def __init__(self, logdir, device="cpu"):
        super().__init__(logdir, device)
        self._load_configs()
        self._load_model()
        self._setup_configs()
        
        # RT policy expects 49-dimensional observations:
        # - Commands: 4 dims (goal segment endpoints: x_l, y_l, x_r, y_r)  
        # - Proprioceptive: 42 dims (ang_vel=3, proj_g=3, joint_pos=12, joint_vel=12, actions=12)
        # - Privileged linear velocity: 3 dims
        self.obs_dim = 49
        self.action_dim = 12
        
        # Required for policy warmup
        self.warm_up_iter = 10

    def _load_configs(self):
        """Load configuration files"""
        env_yaml_path = osp.join(self.logdir, "params", "env.yaml")
        agent_yaml_path = osp.join(self.logdir, "params", "agent.yaml")
        
        assert osp.exists(env_yaml_path), f"Environment config not found at {env_yaml_path}"
        assert osp.exists(agent_yaml_path), f"Agent config not found at {agent_yaml_path}"
        
        with open(env_yaml_path, 'r') as f:
            self.env_config = yaml.load(f, Loader=yaml.UnsafeLoader)
        
        with open(agent_yaml_path, 'r') as f:
            self.agent_config = yaml.load(f, Loader=yaml.UnsafeLoader)

    def _load_model(self):
        """Load the JIT-compiled policy"""
        policy_path = osp.join(self.logdir, "policy.pt")
        assert osp.exists(policy_path), f"Policy file not found at {policy_path}"
        
        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()
        
        print(f"Loaded RT policy from {policy_path}")

    def _setup_configs(self):
        """Setup configurations for the handler"""
        
        # Joint names in the order expected by RT policy
        joint_names = [
            'FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint',
            'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 
            'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint'
        ]
        
        self.joint_map = get_joint_map_from_names(joint_names)
        
        # Default joint positions from env config
        default_joint_pos_dict = {
            'FL_hip_joint': 0.1, 'FR_hip_joint': -0.1, 
            'RL_hip_joint': 0.1, 'RR_hip_joint': -0.1,
            'FL_thigh_joint': 0.8, 'FR_thigh_joint': 0.8,
            'RL_thigh_joint': 1.0, 'RR_thigh_joint': 1.0,
            'FL_calf_joint': -1.5, 'FR_calf_joint': -1.5,
            'RL_calf_joint': -1.5, 'RR_calf_joint': -1.5
        }
        self.default_joint_pos = parse_default_joint_pos_dict(default_joint_pos_dict, joint_names)
        
        # Control parameters from env config
        self.kp = self.env_config["scene"]["robot"]["actuators"]["base_legs"]["stiffness"]  # 25.0
        self.kd = self.env_config["scene"]["robot"]["actuators"]["base_legs"]["damping"]    # 0.5
        self.action_scale = self.env_config["action_scale"]  # 0.25
        
        # Observation and action clipping
        self.clip_obs = 100.0
        self.clip_actions = 100.0

    def get_action(self):
        """Get action from policy given current observations"""
        assert self.handler is not None, "Handler not set"
        
        obs = self._collect_observations()
        
        with torch.no_grad():
            action = self.policy(obs)
        
        return action

    def _collect_observations(self):
        """Collect observations in the format expected by RT policy"""
        
        # Get raw observations from handler
        ang_vel = self.handler.get_ang_vel_obs()              # 3 dims
        proj_g = self._get_projected_gravity()                # 3 dims - gravity in base frame
        joint_pos = self.handler.get_dof_pos_obs()            # 12 dims  
        joint_vel = self.handler.get_dof_vel_obs()            # 12 dims
        last_actions = self.handler.get_last_actions_obs()    # 12 dims
        commands = self._get_rt_commands()                    # 4 dims
        priv_lin_vel = self.handler.get_base_lin_vel_obs()    # 3 dims (privileged)
        
        # Apply scaling as per RT environment configuration
        ang_vel_scaled = ang_vel * 0.25
        proj_g_scaled = proj_g * 0.1  
        joint_pos_scaled = joint_pos * 1.0
        joint_vel_scaled = joint_vel * 0.05
        
        # Concatenate observations: commands + scaled proprioception + privileged velocity
        obs = torch.cat([
            commands,           # 4
            ang_vel_scaled,     # 3
            proj_g_scaled,      # 3
            joint_pos_scaled,   # 12
            joint_vel_scaled,   # 12
            last_actions,       # 12
            priv_lin_vel        # 3
        ], dim=-1)  # Total: 49 dims
        
        assert obs.shape[-1] == self.obs_dim, f"Observation dimension mismatch: {obs.shape[-1]} vs {self.obs_dim}"
        
        return obs

    def _get_rt_commands(self):
        """Get RT-style commands (goal segment endpoints)"""
        # RT uses goal segment commands with 4 dimensions: x_l, y_l, x_r, y_r
        # Fixed goal segment in initial robot frame: 1m forward, 0.8m wide
        # [left_x, left_y, right_x, right_y] = [1.0, 0.4, 1.0, -0.4]
        
        # Fixed goal in initial robot frame (stationary goal)
        goal_initial = torch.tensor([[1.0, 0.4, 1.0, -0.4]], 
                                   device=self.device, dtype=torch.float)
        
        # Get current robot position relative to initial position
        translation = self.handler.get_translation()  # (1, 3) [x, y, z] in initial robot frame
        
        # Transform goal from initial frame to current robot frame
        # Subtract robot's movement from the goal coordinates
        goal_current = goal_initial.clone()
        goal_current[:, [0, 2]] -= translation[:, 0:1]  # Subtract x translation from both endpoints
        goal_current[:, [1, 3]] -= translation[:, 1:2]  # Subtract y translation from both endpoints
        
        return goal_current

    def _get_projected_gravity(self):
        """Get gravity vector projected into robot's base frame"""
        # World gravity vector (pointing down)
        gravity_world = torch.tensor([[0.0, 0.0, -1.0]], device=self.device, dtype=torch.float)
        
        # Get base orientation quaternion
        base_quat = self.handler.get_base_quat_obs()  # (1, 4) in [x,y,z,w] format
        
        # Project gravity into base frame using inverse rotation
        proj_gravity = quat_rotate_inverse(base_quat, gravity_world)
        
        return proj_gravity

    def get_translation_config(self) -> bool:
        """RT policy needs translation tracking to compute commands relative to initial frame"""
        return True
