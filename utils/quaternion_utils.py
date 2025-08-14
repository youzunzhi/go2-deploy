"""
Quaternion utility functions for coordinate transformations.

This module provides PyTorch-based quaternion operations commonly used
in robotics applications, particularly for coordinate frame transformations.
All quaternions are expected to be in [x, y, z, w] format.
"""

import math
import torch


@torch.jit.script  # type: ignore
def copysign(a: float, b: torch.Tensor) -> torch.Tensor:
    """Copy sign of b to magnitude of a."""
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])  # type: ignore
    return torch.abs(a) * torch.sign(b)  # type: ignore


def wrap_to_pi(angle):
    """Wrap angle to [-pi, pi] range."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


@torch.jit.script  # type: ignore
def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Normalize a tensor along the last dimension."""
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)


@torch.jit.script  # type: ignore
def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v from world frame into the frame of quaternion q (inverse rotation).
    
    Args:
        q: Quaternion tensor (B,4) in [x,y,z,w] order
        v: Vector tensor (B,3) to rotate
        
    Returns:
        torch.Tensor: Rotated vector (B,3)
        
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
def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply quaternion rotation to vector.
    
    Args:
        q: Quaternion tensor (B,4) in [x,y,z,w] order
        v: Vector tensor (B,3) to rotate
        
    Returns:
        torch.Tensor: Rotated vector (B,3)
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
    
    Args:
        quat: Full quaternion tensor (B,4) in [x,y,z,w] order
        
    Returns:
        torch.Tensor: Yaw-only quaternion (B,4) in [x,y,z,w] order
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


@torch.jit.script  # type: ignore
def get_euler_xyz(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert quaternion to Euler angles (roll, pitch, yaw).

    Args:
        q: Quaternion tensor (B,4) in [x,y,z,w] order

    Returns:
        tuple: (roll, pitch, yaw) tensors
    """
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = 1.0 - 2.0 * (q[:, qx] * q[:, qx] + q[:, qy] * q[:, qy])
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(
        math.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = 1.0 - 2.0 * (q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz])
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw
