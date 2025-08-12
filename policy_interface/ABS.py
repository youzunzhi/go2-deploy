import os
import math
import torch
from .base import BasePolicyInterface
from utils import get_joint_map_from_names, parse_default_joint_pos_dict


def wrap_to_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


@torch.jit.script  # type: ignore
def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v from world frame into the frame of quaternion q (inverse rotation).
    q is (B,4) in [x,y,z,w] order; v is (B,3). Returns (B,3).
    This matches the Isaac Gym implementation and training behavior.
    """
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


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
        ang_vel = self.handler.get_ang_vel_obs() * self.obs_scales["ang_vel"]  # (1,3)
        base_rpy = self.handler.get_base_rpy_obs()  # (1,3): roll, pitch, yaw
        dof_pos = self.handler.get_dof_pos_obs() * self.obs_scales["dof_pos"]  # (1,12)
        dof_vel = self.handler.get_dof_vel_obs() * self.obs_scales["dof_vel"]  # (1,12)
        last_actions = self.handler.get_last_actions_obs()  # (1,12)
        contact = self.handler.get_contact_filt_obs() * 2.0  # -> [-1, 1] like training

        # Projected gravity in base frame via quaternion rotate-inverse (reads quaternion from IMU)
        # Prefer reading base quaternion directly from handler for fidelity
        base_quat = self.handler.get_base_quat_obs()  # (1,4) xyzw
        gravity_vec = torch.tensor([[0.0, 0.0, -1.0]], device=self.device, dtype=torch.float32)
        proj_gravity = quat_rotate_inverse(base_quat, gravity_vec)

        # Commands from goal and translation (position difference in base-yaw frame)
        translation = self.handler.get_translation()  # (1,3) current position w.r.t. start, meters
        pos_diff_world = self.goal_pose[:, :3] - translation  # (1,3)
        # rotate by -yaw (only yaw component as in training yaw_quat())
        c, s = math.cos(yaw), math.sin(yaw)
        cmd_x = c * pos_diff_world[0, 0] + s * pos_diff_world[0, 1]
        cmd_y = -s * pos_diff_world[0, 0] + c * pos_diff_world[0, 1]
        # heading residual: desired heading = goal_heading + atan2(dy, dx)
        heading_target = wrap_to_pi(self.goal_pose[0, 2].item() + math.atan2(pos_diff_world[0, 1].item(), pos_diff_world[0, 0].item()))
        cmd_yaw = wrap_to_pi(heading_target - yaw)
        commands = torch.tensor([[cmd_x, cmd_y, cmd_yaw]], device=self.device, dtype=torch.float32)

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
        # Joint names in simulation order
        joint_names = [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
        ]
        self.joint_map = get_joint_map_from_names(joint_names)
        default_joint_pos_dict = {
            'FL_hip_joint': 0.1,
            'RL_hip_joint': 0.1,
            'FR_hip_joint': -0.1,
            'RR_hip_joint': -0.1,
            'FL_thigh_joint': 0.8,
            'RL_thigh_joint': 1.0,
            'FR_thigh_joint': 0.8,
            'RR_thigh_joint': 1.0,
            'FL_calf_joint': -1.5,
            'RL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,
            'RR_calf_joint': -1.5,
        }
        self.default_joint_pos = parse_default_joint_pos_dict(default_joint_pos_dict, joint_names)
        # PD gains and action scaling from Go2PosRoughCfg.control
        self.kp = 25.0
        self.kd = 0.6
        self.action_scale = 0.25
        # Normalization/Clipping from Go2PosRoughCfg.normalization
        self.clip_obs = 100.0
        self.clip_actions = None  # let handler/joint limits ensure safety
        self.obs_scales = {
            "ang_vel": 1.0,
            "dof_pos": 1.0,
            "dof_vel": 0.2,
        }
        # Predefined goal pose [x, y, heading]
        self.goal_pose = torch.tensor([[5.0, 0.0, 0.0]], device=self.device, dtype=torch.float32)

    def _load_model(self):
        # Load TorchScript policy exported from training
        model_path = os.path.join(self.logdir, "policy.jit")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"ABS policy not found at {model_path}. Place TorchScript file as 'policy.jit'.")
        self.policy = torch.jit.load(model_path, map_location=self.device)
        self.policy.eval()

