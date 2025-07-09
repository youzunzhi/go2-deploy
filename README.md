# Go2 Deployment System

Deploy reinforcement learning policies on the Unitree Go2 quadruped robot.

## Quick Start

```bash
# Basic execution
python main.py

# Test without robot movement
python main.py --dryrun

# With specific timing mode
python main.py --timing_mode manual_control
```

## Controller Commands

- **R1**: Sport Mode (built-in Unitree behaviors)
- **R2**: Stand Policy (neural network standing)
- **X**: Locomotion Policy (AI-powered walking)
- **Start**: Emergency stop

## Policy Setup

### Extreme-Parkour-Onboard Policies
Place weights and `config.json` in `weight-and-cfg/EPO/`

### legged-loco Policies (In Development)
Place weights and params in `weight-and-cfg/legged-loco/`

## Dependencies

- ROS2
- PyTorch
- Intel RealSense SDK
- Unitree Go2 SDK

## Safety

- Use `--dryrun` for testing without robot movement
- **Start** button provides emergency stop
- Always test policies in simulation first

