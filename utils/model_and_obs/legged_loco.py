import torch
import os

from utils.model_and_obs import BasePolicyCustomManager


class LeggedLocoCustomManager(BasePolicyCustomManager):
    def __init__(self, logdir, device):
        super().__init__(logdir, device)
        self.load_model()
        
    def get_action(self):
        obs = self.get_obs()
        action = self.policy(obs)
        return action

    def get_obs(self):
        assert self.handler is not None, "Handler is not set"
        ang_vel = self.handler._get_ang_vel_obs()
        base_rpy = self.handler._get_base_rpy_obs()
        dof_pos = self.handler._get_dof_pos_obs()
        dof_vel = self.handler._get_dof_vel_obs()
        last_actions = self.handler._get_last_actions_obs()
        
        commands = torch.tensor([[0.4, 0., 0.]], device=self.device, dtype=torch.float32)

        obs = torch.cat([ang_vel, base_rpy, commands, dof_pos, dof_vel, last_actions], dim=-1)

        return obs
        
    def load_model(self):
        model_path = os.path.join(self.logdir, "policy.jit")
        model = torch.jit.load(model_path, map_location=self.device)  # type: ignore
        model.eval()

        self.policy = model.actor_critic.act_inference
