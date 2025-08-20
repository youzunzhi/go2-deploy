# Go2 Deployment Framework

A unified deployment framework for reinforcement learning policies on the Unitree Go2 quadruped robot. This framework enables seamless integration and deployment of policies from various training environments with minimal configuration changes.

## Features

- **Multi-Framework Support**: Deploy policies from legged_gym, IsaacLab, and custom training environments
- **Policy Interface Abstraction**: Clean abstraction layer for different policy types and configurations
- **Vision System Integration**: Non-blocking depth camera capture with Intel RealSense D435i
- **ROS2 Communication**: Full ROS2 integration for robot control and sensor data
- **Safety Systems**: Hardware limits, emergency controls, and fail-fast error handling
- **50Hz Control Loop**: Consistent control frequency matching simulation training
- **Mode Management**: Seamless switching between sport mode, stand policy, and locomotion policy

## Supported Policies

### ✅ Extreme-Parkour-Onboard (EPO)
- **Training Framework**: legged_gym
- **Features**: Advanced parkour locomotion with visual-motor coordination
- **Sensors**: Intel RealSense D435i depth images + proprioceptive observations
- **Status**: Fully supported and tested

### ✅ legged-loco Base Policy
- **Training Framework**: IsaacLab
- **Features**: Base locomotion without vision, 9-step history, 50Hz control
- **Sensors**: Proprioceptive observations only
- **Status**: Fully supported and tested

### ✅ ABS (Position-only Control)
- **Training Framework**: legged_gym
- **Features**: Position-only control policy
- **Sensors**: Proprioceptive observations only
- **Status**: Fully supported and tested

## Quick Start

### Running Existing Policies

```bash
# Extreme-Parkour-Onboard policy (vision-based parkour)
python main.py --logdir weight-and-cfg/EPO

# legged-loco base policy (proprioceptive locomotion)
python main.py --logdir weight-and-cfg/legged-loco-base

# ABS position-only policy
python main.py --logdir weight-and-cfg/ABS

# Debug mode (no robot movement)
python main.py --logdir <policy_path> --dryrun

# Specify compute device
python main.py --logdir <policy_path> --device cuda  # or cpu
```

### Controller Input

#### Safety Controls
- **SELECT**: Emergency safe exit - immediately turns off all motors and exits the program

#### Operational Controls
- **L1**: Switch from sport mode to stand policy
- **Y**: Switch from stand policy to locomotion policy  
- **L2**: Switch back to sport mode from any mode
- **R1**: Stand up (in sport mode)
- **R2**: Sit down (in sport mode)
- **X**: Balance stand (in sport mode)

## Custom Policy Integration

### 1. Create Policy Interface

Create a new file in `policy_interface/` inheriting from `BasePolicyInterface`:

```python
from policy_interface.base import BasePolicyInterface

class YourPolicyInterface(BasePolicyInterface):
    def __init__(self, logdir, device="cpu"):
        # Load policy weights and configurations
        pass
    
    def get_configs_for_handler(self):
        # Return handler configuration tuple
        return (joint_map, default_joint_pos, kp, kd, action_scale, clip_obs, clip_actions)
    
    def set_handler(self, handler):
        self.handler = handler
    
    def get_action(self):
        # Collect observations and compute actions
        obs = self._collect_observations()
        return self.policy(obs)
```

### 2. Available Observations

The handler provides these observation methods:

- `get_xyyaw_command()`: Command velocities (x, y, yaw)
- `get_ang_vel_obs()`: Angular velocities in base frame
- `get_base_rpy_obs()`: Roll, pitch, yaw of the base
- `get_base_quat_obs()`: Base orientation quaternion
- `get_dof_pos_obs()`: Joint positions (normalized)
- `get_dof_vel_obs()`: Joint velocities (normalized)
- `get_last_actions_obs()`: Previous action commands
- `get_contact_filt_obs()`: Filtered foot contact states
- `get_depth_image()`: Depth image tensor from RealSense camera (if enabled)
- `get_translation()`: Robot translation in odometry frame (if enabled)
- `get_translation_world_frame()`: Robot translation in world frame (if enabled)

### 3. Provide Policy Files

Create directory structure under `weight-and-cfg/`:

```
weight-and-cfg/
└── your-policy/
    ├── policy.jit      # PyTorch JIT traced policy
    └── params/         # Optional config files
        ├── agent.yaml
        └── env.yaml
```

### 4. Register in Factory

Update `policy_interface/__init__.py`:

```python
def get_policy_interface(logdir, device):
    # ... existing code ...
    elif "your-policy" in logdir:
        from policy_interface.your_policy import YourPolicyInterface
        return YourPolicyInterface(logdir, device)
```

## Architecture

### Core Components

- **`main.py`**: Main runner orchestrating the control loop
- **`go2_ros2_handler.py`**: ROS2 communication and observation provider
- **`depth_publisher.py`**: Non-blocking depth image capture system
- **`policy_interface/`**: Policy abstraction system with factory pattern
- **`utils/`**: Utility modules for configuration and hardware management

### Control Flow

1. **Initialization**: Load policy interface and configure handler
2. **Sensor Setup**: Start depth publisher if vision is enabled
3. **ROS2 Setup**: Initialize communication and subscribers
4. **Control Loop**: 50Hz main loop with mode management
5. **Safety Monitoring**: Continuous hardware limit and emergency control checking

## Dependencies

- **ROS2**: Robot communication and control
- **PyTorch**: Deep learning inference
- **OpenCV**: Computer vision processing
- **Intel RealSense SDK**: Depth camera integration
- **Unitree SDK**: Robot hardware interface
- **rsl_rl**: Reinforcement learning utilities

## Known Limitations

- LiDAR heightmap data unavailable when sport mode is disabled (affects legged-loco vision policies)
- Intel RealSense D435i required for vision-based policies

## Safety Notice

This system controls a physical robot. Always ensure proper safety measures:
- Use SELECT button for emergency shutdown
- Test in dryrun mode first
- Maintain clear workspace around the robot
- Have physical emergency stop readily available

## Acknowledgments

This framework is built upon and inspired by the excellent work at [Extreme-Parkour-Onboard](https://github.com/change-every/Extreme-Parkour-Onboard). Special thanks to the original authors for their contributions to legged robot deployment systems.

## License

See LICENSE file for details.