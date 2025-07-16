import torch

class LeggedLocoObs:
    def __init__(self, device):
        self.device = device

    def get_obs(self, ang_vel, base_rpy, dof_pos, dof_vel, last_actions):
        commands = torch.tensor([[0.4, 0., 0.]], device=self.device, dtype=torch.float32)

        obs = torch.cat([ang_vel, base_rpy, commands, dof_pos, dof_vel, last_actions], dim=-1)

        return obs