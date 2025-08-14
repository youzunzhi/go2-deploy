import os
import math
import torch
from collections import OrderedDict
import json
import os.path as osp

from .base import BasePolicyInterface
from utils import get_joint_map_from_names, parse_default_joint_pos_dict
from utils.quaternion_utils import wrap_to_pi, quat_rotate_inverse, quat_apply, yaw_quat


class ABSPolicyInterface(BasePolicyInterface):
    def __init__(self, logdir, device):
        super().__init__(logdir, device)
        self._load_configs()
        self._load_model()

        self.warm_up_iter = 10

    def get_action(self):
        self.policy_iter_counter += 1
        obs = self._get_obs()
        action = self.policy(obs)
        return action

    def _get_obs(self):
        # Proprioception
        contact = self.handler.get_contact_filt_obs() * 2.0  # -> [-1, 1] like training
        ang_vel = self.handler.get_ang_vel_obs() * self.obs_scales["ang_vel"]  # (1,3)
        dof_pos = self.handler.get_dof_pos_obs() * self.obs_scales["dof_pos"]  # (1,12)
        dof_vel = self.handler.get_dof_vel_obs() * self.obs_scales["dof_vel"]  # (1,12)
        last_actions = self.handler.get_last_actions_obs()  # (1,12)

        # Projected gravity in base frame via quaternion rotate-inverse (reads quaternion from IMU)
        # Prefer reading base quaternion directly from handler for fidelity
        base_quat = self.handler.get_base_quat_obs()  # (1,4) xyzw
        gravity_vec = torch.tensor([[0.0, 0.0, -1.0]], device=self.device, dtype=torch.float32)
        proj_gravity = quat_rotate_inverse(base_quat, gravity_vec)

        # Commands from goal and translation (position difference in base-yaw frame)
        # This matches the training code in _post_physics_step_callback()
        translation = self.handler.get_translation()  # (1,3) current position w.r.t. start, meters
        pos_diff = self.goal_pose[:, :2] - translation[:, :2]  # (1,2) position difference in world frame (xy only)
        pos_diff_3d = torch.cat([pos_diff, torch.zeros_like(pos_diff[:, :1])], dim=1)  # (1,3) add z=0 for rotation
        commands_xy = quat_rotate_inverse(yaw_quat(base_quat), pos_diff_3d)[:, :2]  # Transform to base-yaw frame
        # Compute heading command
        forward_vec = torch.tensor([[1.0, 0.0, 0.0]], device=self.device, dtype=torch.float32)  # Forward direction
        forward = quat_apply(base_quat, forward_vec)  # Current heading direction in world frame
        current_heading = torch.atan2(forward[:, 1], forward[:, 0])  # Current heading angle
        cmd_yaw = wrap_to_pi(self.heading_target - current_heading[0].item())
        commands = torch.cat([commands_xy, torch.tensor([[cmd_yaw]], device=self.device, dtype=torch.float32)], dim=1)

        # Timer left normalized (no timer in deployment) -> set to constant 0.5
        timer_left = torch.tensor([[0.5]], device=self.device, dtype=torch.float32)

        obs = torch.cat([
            contact,                # 4
            ang_vel,                # 3
            proj_gravity,           # 3
            commands,               # 3
            timer_left,             # 1
            dof_pos,                # 12
            dof_vel,                # 12
            last_actions            # 12
        ], dim=-1)

        # Clip observations
        if self.clip_obs is not None:
            obs = torch.clamp(obs, -self.clip_obs, self.clip_obs)
        return obs

    def _load_configs(self):
        # Configuration adapted from Go2PosRoughCfg (training env), excluding ray2d
        config_path = osp.join(self.logdir, "config.json")
        with open(config_path, "r") as f:
            full_config = json.load(f, object_pairs_hook=OrderedDict)
        env_config = full_config["env_cfg"]
        # Joint names in simulation order
        joint_names = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ]
        self.joint_map = get_joint_map_from_names(joint_names)

        default_joint_pos_dict = env_config["init_state"]["default_joint_angles"]
        self.default_joint_pos = parse_default_joint_pos_dict(default_joint_pos_dict, joint_names)
        self.kp = float(env_config["control"]["stiffness"]["joint"])
        self.kd = float(env_config["control"]["damping"]["joint"])
        self.action_scale = float(env_config["control"]["action_scale"])
        self.clip_obs = env_config["normalization"]["clip_observations"]
        self.clip_actions = env_config["normalization"]["clip_actions"]
        self.obs_scales = env_config["normalization"]["obs_scales"]
        
        self.goal_pose = torch.tensor([[5.0, 0.0, 0.0]], device=self.device, dtype=torch.float32)
        direction_to_goal = math.atan2(self.goal_pose[0, 1].item(), self.goal_pose[0, 0].item())
        self.heading_target = wrap_to_pi(self.goal_pose[0, 2].item() + direction_to_goal)

    def _load_model(self):
        # Load TorchScript policy exported from training
        model_path = os.path.join(self.logdir, "policy_jit.pt")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"ABS policy not found at {model_path}. Place TorchScript file as 'policy_jit.pt'.")
        self.policy = torch.jit.load(model_path, map_location=self.device)
        self.policy.eval()

    def get_translation_config(self) -> bool:
        """ABS policy requires translation capture for goal-based navigation"""
        return True

