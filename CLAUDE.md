# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a robotics deployment system for the Unitree Go2 quadruped robot, combining ROS2 with PyTorch-based deep learning for autonomous locomotion. The system integrates visual perception, proprioception, and neural network inference for real-time robot control.

## Architecture

### Core Components

- **`main.py`** - Primary orchestration system managing the control loop, neural network inference, and mode switching
- **`go2_ros2_node.py`** - ROS2 node handling robot communication, sensor data processing, and low-level motor control
- **`visual_node.py`** - Visual processing pipeline for Intel RealSense depth camera integration
- **`weight-and-cfg/`** - Neural network weights and comprehensive robot configuration

### Control Flow

1. Initialize ROS2 node and robot connections
2. Load policy-specific configurations and weights when switching to neural network modes
3. Run 50Hz control loop (20ms intervals) with timing modes:
   - **ROS Timer**: Standard ROS2-managed timing
   - **Manual Control**: Precise timing for high-performance operation
4. Switch between operational modes based on controller input:
   - **Sport Mode**: Built-in Unitree behaviors (stand, sit, balance)
   - **Stand Policy**: Neural network-based standing with disturbance rejection
   - **Locomotion Policy**: AI-powered walking with visual-motor coordination

### Neural Network Architecture

The system uses multiple specialized networks:
- **StateHistoryEncoder**: Historical state processing for temporal awareness
- **RecurrentDepthBackbone**: Depth perception processing
- **DepthOnlyFCBackbone58x87**: Visual feature extraction from depth images
- Models are loaded as PyTorch JIT (.pt) or standard PyTorch (.pth) files

## Configuration System

The system uses policy-specific configuration files for RL model deployment:
- **`weight-and-cfg/EPO/config.json`** - Configuration for EPO policy training and deployment in simulation
- Contains RL-specific parameters like observation/action spaces, training hyperparameters, and simulation environment settings
- Used specifically for EPO policy weight loading and deployment
- Future expansion planned to support navila-loco configurations and weights for broader policy compatibility

## Development Commands

### Running the System

```bash
# Basic execution
python main.py

# With specific timing mode
python main.py --timing_mode ros_timer  # or manual_control

# Debug mode without robot movement
python main.py --dryrun
```

### Controller Input

- **R1**: Sport mode (built-in behaviors)
- **R2**: Stand policy (neural network standing)
- **X**: Locomotion policy (AI walking)
- **Start**: Emergency stop

## Dependencies

**Core Stack:**
- ROS2 (Robot Operating System 2)
- PyTorch (deep learning inference)
- OpenCV (computer vision)
- NumPy (numerical computations)
- Intel RealSense SDK (depth camera)
- Unitree SDK (robot hardware)
- rsl_rl (reinforcement learning library)

**Architecture Support:**
- x86_64 and aarch64 architectures
- CRC module for reliable communication

## Performance Monitoring

The system includes detailed timing analysis of:
- Proprioception acquisition time
- Visual processing time  
- Neural network inference time
- Command publishing time
- Total control loop time

Use timing logs to optimize performance and identify bottlenecks.

## Safety Features

- **Dryrun mode**: Testing without robot movement
- **Collision detection**: Contact force monitoring
- **Emergency stop**: Controller-based safety shutdown
- **Hardware abstraction**: Safe motor control with PID gains

## Policy Configuration

Configuration files are copied from simulation training projects and not modified in go2-deploy:
- EPO policy configs are copied from EPO training projects to `weight-and-cfg/EPO/`
- Future support planned for navila-loco configs copied to `weight-and-cfg/navila-loco/`
- Config files contain training-specific parameters like action scaling, terrain settings, and reward weights
- Robot behavior modifications should be made in the original training projects, not in the deployment system

## File Structure Notes

- Model weights should be placed in `weight-and-cfg/navila-loco/` for locomotion models
- Configuration changes require restart of the main control loop
- Visual processing parameters are hardcoded in `visual_node.py` for performance
- ROS2 node parameters are configured in `go2_ros2_node.py`