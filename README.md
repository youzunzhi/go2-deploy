# Go2 Deployment System

Deploy reinforcement learning policies on the Unitree Go2 quadruped robot.

## Quick Start

### Put weights and config files in `weight-and-config/`

### Implement configuration loading and parsing
- In `load_configuration` in `main.py`, implement how your config file should be load and parse to get the configs needed in Go2ROS2Node
    - The joint order in simulation should be correctly specified in load_and_parse_configuration by joint_names
    - The control type of your policy output is target position relative to the default joint positions
    - The duration is set to fixed 0.02s, assuming dt=0.005 and decimation=4 in sim

```bash
# Basic execution
python main.py

# Test without robot movement
python main.py --dryrun
```

## Controller Commands

- **R1**: Sport Mode (built-in Unitree behaviors)
- **R2**: Stand Policy (neural network standing)
- **X**: Locomotion Policy (AI-powered walking)
- **Start**: Emergency stop


## Dependencies

- ROS2
- PyTorch
- Intel RealSense SDK
- Unitree Go2 SDK

## Safety

- Use `--dryrun` for testing without robot movement
- **Start** button provides emergency stop
- Always test policies in simulation first