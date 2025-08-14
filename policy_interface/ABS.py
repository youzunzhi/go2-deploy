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
        # get directly from handler
        contact = self.handler.get_contact_filt_obs() * 2.0  # -> [-1, 1] like training
        ang_vel = self.handler.get_ang_vel_obs() * self.obs_scales["ang_vel"]  # (1,3)
        dof_pos = self.handler.get_dof_pos_obs() * self.obs_scales["dof_pos"]  # (1,12)
        dof_vel = self.handler.get_dof_vel_obs() * self.obs_scales["dof_vel"]  # (1,12)
        last_actions = self.handler.get_last_actions_obs()  # (1,12)

        # Projected gravity in base frame via quaternion rotate-inverse (reads quaternion from IMU)
        base_quat = self.handler.get_base_quat_obs()  # (1,4) xyzw
        gravity_vec_w = torch.tensor([[0.0, 0.0, -1.0]], device=self.device, dtype=torch.float32)  # Gravity vector in world frame
        proj_gravity = quat_rotate_inverse(base_quat, gravity_vec_w)    # (1,3) projected gravity in base frame

        # commands_xy = goal xy in initial robot frame - current xy in initial robot frame
        #             = goal xy in initial robot frame - translation
        translation = self.handler.get_translation()  # (1,3) current position w.r.t. start in robot's initial frame
        commands_xy = self.goal_pose[:, :2] - translation[:, :2]  # (1,2) position difference in robot's initial frame (xy only)
        # commands_yaw = goal yaw in initial robot frame - current yaw in initial robot frame
        # Compute heading command - target heading is relative to robot's initial orientation
        forward_vec_b = torch.tensor([[1.0, 0.0, 0.0]], device=self.device, dtype=torch.float32)  # Forward vec in base frame
        cur_heading_vec_w = quat_apply(base_quat, forward_vec_b)  # Current heading direction in world frame
        cur_yaw_w = torch.atan2(cur_heading_vec_w[:, 1], cur_heading_vec_w[:, 0])  # Current heading angle in world frame
        # Get initial heading direction in world frame
        start_heading_vec_w = quat_apply(self.handler.start_quat, forward_vec_b)  # Initial heading in world frame
        start_yaw_w = torch.atan2(start_heading_vec_w[:, 1], start_heading_vec_w[:, 0])  # Initial heading angle
        # Calculate heading relative to initial orientation
        cur_yaw_i = wrap_to_pi(cur_yaw_w[0].item() - start_yaw_w[0].item()) # current yaw in initial base frame
        commands_yaw = wrap_to_pi(self.goal_pose[:, 2] - cur_yaw_i)
        commands = torch.cat([commands_xy, torch.tensor([[commands_yaw]], device=self.device, dtype=torch.float32)], dim=1)

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
        env_config = full_config["env_config"]
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
        
        # Goal pose in robot's initial frame: x=forward, y=left, z=up relative to start pose
        self.goal_pose = torch.tensor([[2.5, 0.0, 0.0]], device=self.device, dtype=torch.float32)

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

