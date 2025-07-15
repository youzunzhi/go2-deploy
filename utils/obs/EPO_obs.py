import torch
import torch.nn as nn

from utils.networks.EPO import load_base_model


class EPO_obs:
    def __init__(self, logdir, device, depth_encoder, estimator, hist_encoder, 
    visual_update_interval, parkour_walk_mode, n_hist_len, n_proprio, obs_scales):
        self.logdir = logdir
        self.device = device
        self.depth_encoder = depth_encoder
        self.estimator = estimator
        self.hist_encoder = hist_encoder
        self.visual_update_interval = visual_update_interval
        if parkour_walk_mode == "parkour":
            self.parkour_walk_tensor = torch.tensor([[1, 0]], device=self.device, dtype=torch.float32)
        elif parkour_walk_mode == "walk":
            self.parkour_walk_tensor = torch.tensor([[0, 1]], device=self.device, dtype=torch.float32)
        self.n_hist_len = n_hist_len
        self.n_proprio = n_proprio
        self.obs_scales = obs_scales

        self.last_depth_image = None
        self.proprio_history_buf = torch.zeros(1, self.n_hist_len, self.n_proprio, device=self.device, dtype=torch.float)
        self.episode_length_buf = torch.zeros(1, device=self.device, dtype=torch.float)

        self.activation = nn.ELU()

    def get_EPO_obs(self, global_counter, depth_image, ang_vel, base_rpy, dof_pos, dof_vel, last_actions, contact):
        proprio = self.get_proprio(ang_vel, base_rpy, dof_pos, dof_vel, last_actions, contact)
        if global_counter % self.visual_update_interval == 0:
            if global_counter == 0:
                last_depth_image = depth_image
            depth_latent_yaw = self.depth_encoder(last_depth_image, proprio)
        self.last_depth_image = depth_image
        # Separate depth features and yaw angle
        depth_latent = depth_latent_yaw[:, :-2]
        yaw = depth_latent_yaw[:, -2:] * 1.5
        
        # Update yaw angle in proprioceptive data
        proprio[:, 6:8] = yaw
        
        # Estimate linear velocity features
        lin_vel_latent = self.estimator(proprio)
        
        # Process historical proprioceptive data
        priv_latent = self.hist_encoder(self.activation, self.proprio_history_buf.view(-1, self.n_hist_len, self.n_proprio))
        
        # Concatenate all features
        obs = torch.cat([proprio, depth_latent, lin_vel_latent, priv_latent], dim=-1)
        
        return obs

    def get_proprio(self, ang_vel, base_rpy, dof_pos, dof_vel, last_actions, contact):
        ang_vel = ang_vel * self.obs_scales["ang_vel"]
        
        imu = base_rpy[:, 0:2]
        
        yaw_info = torch.tensor([[0, 0, 0]], device=self.device, dtype=torch.float32)

        commands = torch.tensor([[0, 0, 0.5]], device=self.device, dtype=torch.float32)
        
        dof_pos = dof_pos * self.obs_scales["dof_pos"]
        dof_vel = dof_vel * self.obs_scales["dof_vel"]
        
        proprio = torch.cat([ang_vel, imu, yaw_info, commands, self.parkour_walk_tensor,
                        dof_pos, dof_vel,
                        last_actions, 
                        contact], dim=-1)

        self.proprio_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([proprio] * self.n_hist_len, dim=1),
            torch.cat([
                self.proprio_history_buf[:, 1:],
                proprio.unsqueeze(1)
            ], dim=1)
        )

        self.episode_length_buf += 1

        return proprio