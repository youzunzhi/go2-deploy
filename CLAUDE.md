# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a robotics deployment system for the Unitree Go2 quadruped robot that deploys reinforcement learning policies from multiple training environments. Originally based on Extreme-Parkour-Onboard, this system is being refactored to support policies from both legged_gym (Extreme-Parkour-Onboard) and IsaacLab (legged-loco) training environments.

## Current Development Goals

**Primary Objective**: Create a unified deployment system that can load and run RL policies from different training environments with minimal configuration changes.

**Immediate Focus**: 
- Deploy locomotion policies from `logs/rsl_rl/go2_base/2025-07-03_21-32-44_XXX/` (legged-loco/IsaacLab)
- Maintain compatibility with existing Extreme-Parkour-Onboard policies
- Handle differences in configuration formats and observation structures between training environments

**Key Challenge**: Different training environments produce policies with incompatible configuration methods and observation structures (e.g., `clip_actions` exists in Extreme-Parkour-Onboard configs but not in legged-loco configs).

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

The system supports multiple neural network architectures depending on the policy source:
- **StateHistoryEncoder**: Historical state processing for temporal awareness
- **RecurrentDepthBackbone**: Depth perception processing  
- **DepthOnlyFCBackbone58x87**: Visual feature extraction from depth images
- Models are loaded as PyTorch JIT (.pt) or standard PyTorch (.pth) files
- **Multi-policy support**: Architecture detection and loading based on policy source

## Policy Sources and Configuration System

**IMPORTANT**: Policy source repositories (legged-loco, Extreme-Parkour-Onboard) are READ-ONLY. Never modify code in these repositories - only copy weights and configurations to go2-deploy.

### Supported Policy Sources

1. **Extreme-Parkour-Onboard** (legged_gym-based)
   - Current configuration: `weight-and-cfg/EPO/config.json`
   - Contains parameters like `clip_actions`, action scaling, terrain settings
   - Fully supported and tested

2. **legged-loco** (IsaacLab-based) - **IN DEVELOPMENT**
   - Target policies: `logs/rsl_rl/go2_base/2025-07-03_21-32-44_XXX/`
   - Configuration stored in `params/` subdirectory
   - Different configuration structure than Extreme-Parkour-Onboard
   - Main challenge: Configuration mapping between training environments

### Configuration Challenges

- **Format Differences**: Each training environment uses different configuration file structures
- **Missing Parameters**: Some parameters exist in one environment but not another (e.g., `clip_actions`)
- **Observation Structures**: Different training environments may have different observation space definitions
- **Action Scaling**: Different approaches to action normalization and scaling

### Directory Structure for Multi-Policy Support

```
weight-and-cfg/
├── EPO/                    # Extreme-Parkour-Onboard policies
│   ├── config.json
│   └── *.pt/*.pth
├── legged-loco/           # IsaacLab policies (planned)
│   ├── params/            # Configuration files
│   └── *.pt/*.pth         # Model weights
└── [future-sources]/      # Additional training environments
```

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

## Development Status and Next Steps

### Current Implementation Status
- ✅ Basic deployment system working with Extreme-Parkour-Onboard policies
- ✅ ROS2 integration and robot communication
- ✅ Visual processing pipeline
- 🔄 **IN PROGRESS**: Configuration loading system refactoring
- ❌ **TODO**: legged-loco policy support
- ❌ **TODO**: Unified configuration interface

### Immediate Development Tasks
1. **Configuration System Refactoring**: Create unified interface for loading configurations from different training environments
2. **Policy Loader Abstraction**: Abstract policy loading to handle different weight formats and observation structures
3. **Configuration Mapping**: Implement translation layer between different configuration formats
4. **Testing Framework**: Ensure new policies work correctly without breaking existing functionality

### Code Organization Principles
- **Read-only external repositories**: Never modify legged-loco or Extreme-Parkour-Onboard code
- **Copy, don't link**: Copy necessary weights and configurations to go2-deploy structure
- **Abstraction layers**: Create interfaces that hide differences between training environments
- **Backward compatibility**: Maintain support for existing Extreme-Parkour-Onboard policies

## File Structure and Development Notes

### Policy Organization
- **Extreme-Parkour-Onboard**: `weight-and-cfg/EPO/` (current)
- **legged-loco**: `weight-and-cfg/legged-loco/` (planned)
- **Future policies**: `weight-and-cfg/[source-name]/`

### Development Guidelines
- Configuration changes require restart of the main control loop
- Visual processing parameters are hardcoded in `visual_node.py` for performance
- ROS2 node parameters are configured in `go2_ros2_node.py`
- **Never modify external repositories**: legged-loco and Extreme-Parkour-Onboard are read-only
- **Copy files only**: Copy weights and configs to go2-deploy, don't create symlinks

### Refactoring Focus Areas
- `main.py`: Configuration loading and policy switching logic
- Policy loading abstraction for different training environments
- Configuration format translation and validation
- Observation structure handling for different policy types