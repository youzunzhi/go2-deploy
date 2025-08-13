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


@torch.jit.script  # type: ignore
def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Normalize a tensor along the last dimension."""
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)


@torch.jit.script  # type: ignore
def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply quaternion rotation to vector.
    q is (B,4) in [x,y,z,w] order; v is (B,3). Returns (B,3).
    """
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


@torch.jit.script  # type: ignore
def yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    """Extract yaw-only quaternion from full quaternion.
    quat is (B,4) in [x,y,z,w] order. Returns (B,4).
    """
    quat_yaw = quat.clone().view(-1, 4)
    qx = quat_yaw[:, 0]
    qy = quat_yaw[:, 1]
    qz = quat_yaw[:, 2]
    qw = quat_yaw[:, 3]
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    quat_yaw[:, :2] = 0.0
    quat_yaw[:, 2] = torch.sin(yaw / 2)
    quat_yaw[:, 3] = torch.cos(yaw / 2)
    quat_yaw = normalize(quat_yaw)
    return quat_yaw


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
        # Joint names in simulation order
        joint_names = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ]
        self.joint_map = get_joint_map_from_names(joint_names)
        default_joint_pos_dict = {
            'FL_hip_joint': 0.1,
            'FL_thigh_joint': 0.8,
            'FL_calf_joint': -1.5,
            'FR_hip_joint': -0.1,
            'FR_thigh_joint': 0.8,
            'FR_calf_joint': -1.5,
            'RL_hip_joint': 0.1,
            'RL_thigh_joint': 1.0,
            'RL_calf_joint': -1.5,
            'RR_hip_joint': -0.1,
            'RR_thigh_joint': 1.0,
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
        # Set default goal and compute heading target
        self.goal_pose = torch.tensor([[5.0, 0.0, 0.0]], device=self.device, dtype=torch.float32)
        direction_to_goal = math.atan2(self.goal_pose[0, 1].item(), self.goal_pose[0, 0].item())
        self.heading_target = wrap_to_pi(self.goal_pose[0, 2].item() + direction_to_goal)

    def _load_model(self):
        # Load TorchScript policy exported from training
        model_path = os.path.join(self.logdir, "policy.jit")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"ABS policy not found at {model_path}. Place TorchScript file as 'policy.jit'.")
        self.policy = torch.jit.load(model_path, map_location=self.device)
        self.policy.eval()

