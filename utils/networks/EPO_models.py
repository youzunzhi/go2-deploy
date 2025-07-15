import os
import torch
import torch.nn as nn

def load_base_model(logdir, device):
    """
    Load base model (JIT format)
    
    Args:
        logdir: Model file directory
        device: Computing device (cuda/cpu)
        
    Returns:
        base_model: Loaded base model
        estimator: Speed estimator
        hist_encoder: History encoder
        actor: Action generator
    """
    base_model_name = 'base_jit.pt'
    base_model_path = os.path.join(logdir, base_model_name)
    
    # Load base model in JIT format
    base_model = torch.jit.load(base_model_path, map_location=device)
    base_model.eval()
    
    # Extract model components
    estimator = base_model.estimator.estimator
    hist_encoder = base_model.actor.history_encoder
    actor = base_model.actor.actor_backbone
    
    return base_model, estimator, hist_encoder, actor


class DepthOnlyFCBackbone58x87(nn.Module):
    def __init__(self, prop_dim, scandots_output_dim, hidden_state_dim, output_activation=None, num_frames=1):
        super().__init__()

        self.num_frames = num_frames
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            # [1, 58, 87]
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            # [32, 54, 83]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 27, 41]
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            activation,
            nn.Flatten(),
            # [32, 25, 39]
            nn.Linear(64 * 25 * 39, 128),
            activation,
            nn.Linear(128, scandots_output_dim)
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

    def forward(self, images: torch.Tensor):
        images_compressed = self.image_compression(images.unsqueeze(1))
        latent = self.output_activation(images_compressed)

        return latent


class RecurrentDepthBackbone(nn.Module):
    def __init__(self, base_backbone, env_cfg) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = base_backbone
        if env_cfg == None:
            self.combination_mlp = nn.Sequential(
                                    nn.Linear(32 + 53, 128),
                                    activation,
                                    nn.Linear(128, 32)
                                )
        else:
            self.combination_mlp = nn.Sequential(
                                        nn.Linear(32 + env_cfg.env.n_proprio, 128),
                                        activation,
                                        nn.Linear(128, 32)
                                    )
        self.rnn = nn.GRU(input_size=32, hidden_size=512, batch_first=True)
        # 为什么要 32 + 2, 根据之后的代码，scandot latent 是 32 维的
        # 但 backbone 代码明确写了，和 scandot latent 维度是一样的
        # 所以这个有 recurrent 的 depth backbone 另有他用
        self.output_mlp = nn.Sequential(
                                nn.Linear(512, 32+2),
                                last_activation
                            )
        self.hidden_states = None

    def forward(self, depth_image, proprioception):
        # 先处理深度图像，得到一个基础的特征表示，32维的 latent 数据
        depth_image = self.base_backbone(depth_image)
        # 再把 proprioception 和基础的图像 latent 拼接在一起，传入一个MLP处理 
        depth_latent = self.combination_mlp(torch.cat((depth_image, proprioception), dim=-1))
        # 最后通过 RNN 得到最终的 latent 表示
        depth_latent, self.hidden_states = self.rnn(depth_latent[:, None, :], self.hidden_states)
        # 根据 output 全连接网络来看，输出维度应该是 34？
        depth_latent = self.output_mlp(depth_latent.squeeze(1))
        
        return depth_latent

    def detach_hidden_states(self):
        self.hidden_states = self.hidden_states.detach().clone()


def load_vision_model(logdir, device):
    """
    Load vision model (depth encoder)
    
    Args:
        logdir: Model file directory
        device: Computing device
        
    Returns:
        depth_encoder: Depth encoder model
    """
    vision_model_name = 'vision_weight.pt'
    vision_model_path = os.path.join(logdir, vision_model_name)
    
    # Load vision model weights
    vision_model = torch.load(vision_model_path, map_location=device)
    
    # Create depth encoder
    depth_backbone = DepthOnlyFCBackbone58x87(None, 32, 512)
    depth_encoder = RecurrentDepthBackbone(depth_backbone, None).to(device)
    
    # Load pre-trained weights
    depth_encoder.load_state_dict(vision_model['depth_encoder_state_dict'])
    depth_encoder.to(device)
    depth_encoder.eval()
    
    return depth_encoder