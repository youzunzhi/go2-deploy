# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a robotics deployment system for the Unitree Go2 quadruped robot designed to deploy reinforcement learning policies from multiple training environments. The system has been refactored to support policies from both legged_gym (Extreme-Parkour-Onboard) and IsaacLab (legged-loco) training environments through a policy interface abstraction.

## Current Development Goals

**Primary Objective**: Create a unified deployment system that can load and run RL policies from different training environments with minimal configuration changes.

**Current Focus**: 
- Testing the Go2 base policy from ~/legged-loco/logs/rsl_rl/go2_base/2025-07-03_21-32-44_XXX/exported/policy.jit
- Validating legged-loco policy interface integration between simulation training and hardware deployment
- Ensuring consistent joint mappings, control parameters, and observation spaces for legged-loco policies
- Real-world performance testing and validation of Go2 base locomotion

## Architecture

### Core Components

- **`main.py`** - Main runner class (`Go2Runner`) orchestrating the control loop and system initialization
- **`go2_ros2_handler.py`** - ROS2 handler managing robot communication, sensor data processing, motor control, and observation collection
- **`policy_interface/legged_loco.py`** - IsaacLab/legged-loco policy implementation (CURRENT TESTING FOCUS)
- **`policy_interface/base.py`** - Abstract base class for policy interfaces
- **`policy_interface/__init__.py`** - Factory function for policy interface selection
- **`utils/control_mode_manager.py`** - State management for robot operational modes (sport/stand/locomotion)
- **`utils/config.py`** - Configuration utilities and joint mapping
- **`utils/hardware.py`** - Hardware-specific constants and limits

### Control Flow

1. **Initialization**: `Go2Runner` creates policy interface, handler, and sport mode manager
2. **Policy Interface**: Detects and loads appropriate policy based on logdir path
3. **Configuration Loading**: Policy interface provides handler configuration (joint maps, PID gains, scaling, etc.)
4. **ROS Setup**: Handler initializes ROS2 publishers, subscribers, and communication
5. **Control Loop**: 50Hz main loop for consistent control frequency matching simulation training
6. **Mode Management**: Sport mode manager handles state transitions based on controller input:
   - **Sport Mode**: Built-in Unitree behaviors (stand, sit, balance)
   - **Stand Policy**: Neural network-based standing with disturbance rejection
   - **Locomotion Policy**: AI-powered walking with visual-motor coordination

## Policy Interface System

The policy interface abstraction handles the complexity of different training environments:

### Interface Structure
```python
class BasePolicyInterface:
    def get_configs_for_handler() -> (joint_map, default_joint_pos, kp, kd, action_scale, clip_obs, clip_actions)
    def set_handler(handler)
    def get_action() -> action_tensor
```

### Current Policy Source

**legged-loco (IsaacLab)** - CURRENT TESTING FOCUS
- Source: `~/legged-loco/logs/rsl_rl/go2_base/2025-07-03_21-32-44_XXX/exported/policy.jit`
- Configuration: `weight-and-cfg/legged-loco/params/`
- Features: Go2 base locomotion (no vision), 9-step history, 50Hz control
- Training details: RSL-RL with PPO, trained in Isaac Lab simulation
- Critical requirement: Consistent joint mappings and control parameters between simulation and hardware

### Directory Structure
```
go2-deploy/
├── main.py                 # Main runner and entry point
├── go2_ros2_handler.py     # ROS2 handler and robot control
├── go2_controller.py       # Controller utilities
├── policy_interface/       # Policy abstraction system
│   ├── __init__.py         # Factory function for policy selection
│   ├── base.py             # Abstract base class
│   └── legged_loco.py      # IsaacLab implementation (CURRENT TESTING FOCUS)
├── utils/                  # Utility modules
│   ├── __init__.py
│   ├── config.py           # Configuration utilities and joint mapping
│   ├── hardware.py         # Hardware constants and limits
│   └── control_mode_manager.py # Robot mode state management
├── weight-and-cfg/         # Neural network weights and configurations
│   └── legged-loco/        # IsaacLab policies (CURRENT TESTING FOCUS)
│       ├── params/
│       │   ├── agent.yaml
│       │   └── env.yaml
│       └── policy.jit      # Copy from ~/legged-loco training outputs
├── aarch64/                # ARM64 architecture binaries
│   └── crc_module.so
├── x86/                    # x86_64 architecture binaries
│   └── crc_module.so
├── CLAUDE.md               # Project documentation for Claude Code
├── README.md               # Project readme
├── QUICKSTART.md           # Quick start guide
└── LICENSE                 # License file

External Dependencies:
├── ~/legged-loco/          # Training repository (read-only)
│   └── logs/rsl_rl/go2_base/2025-07-03_21-32-44_XXX/exported/policy.jit
```

## Development Commands

### Running the System

```bash
# Run with legged-loco policy (CURRENT TESTING FOCUS)
python main.py --logdir weight-and-cfg/legged-loco

# Debug mode without robot movement
python main.py --logdir weight-and-cfg/legged-loco --dryrun

# Specify device
python main.py --logdir weight-and-cfg/legged-loco --device cuda  # or cpu
```

### Controller Input

- **L1**: Switch from sport mode to stand policy
- **Y**: Switch from stand policy to locomotion policy  
- **L2**: Switch back to sport mode from any mode
- **R1**: Stand up (in sport mode)
- **R2**: Sit down (in sport mode)
- **X**: Balance stand (in sport mode)

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

## Safety Features

- **Dryrun mode**: Testing without robot movement
- **Joint limit clipping**: Enforces hardware joint position limits
- **Torque limit clipping**: Prevents excessive motor torques
- **Contact force monitoring**: Foot contact detection for safety
- **Emergency modes**: Controller-based safety shutdown and mode switching
- **Hardware abstraction**: Safe motor control with configurable PID gains

## Current Development Status

### Implementation Status
- Multi-policy deployment system structure in place
- Policy interface abstraction implemented
- Configuration management system refactored
- ROS2 integration and robot communication
- Visual processing pipeline
- Sport mode management
- Safety systems and motor control

### Immediate Tasks
1. **Policy Integration Testing**: Validate legged-loco policy loading and execution
2. **Joint Mapping Verification**: Ensure consistent joint order between simulation and hardware
3. **Control Parameter Validation**: Verify 50Hz control frequency and observation history handling
4. **Real Robot Testing**: Test Go2 base locomotion performance on hardware
5. **Performance Analysis**: Monitor policy execution timing and robot response

## File Structure and Development Guidelines

### Core Files
- **`main.py`**: Main runner and argument parsing
- **`go2_ros2_handler.py`**: ROS2 communication and robot control
- **`policy_interface/legged_loco.py`**: legged-loco policy implementation *(CURRENT TESTING FOCUS)*
- **`utils/`**: Utility modules (config, hardware, sport mode management)

### Development Principles
- **Read-only external repositories**: Never modify legged-loco training code in ~/legged-loco
- **Simulation consistency**: Maintain exact consistency between simulation training parameters and hardware deployment
- **Clean abstractions**: Maintain clear separation between policy logic and robot control
- **Safety first**: Always maintain hardware safety limits and emergency controls

### Configuration Management
- **Do not modify configuration files**: Configuration files in weight-and-cfg/ are copied from simulation training and should remain unchanged
- **Policy-specific configs**: Each policy interface handles its own configuration format
- **Automatic detection**: System automatically selects appropriate policy interface based on logdir path

### Testing Goals for legged-loco Integration
The current focus is testing the Go2 base policy deployment:
- Validate policy loading from ~/legged-loco training outputs
- Ensure joint mappings match between IsaacLab simulation and Unitree hardware
- Verify observation space consistency (9-step history, joint positions, velocities)
- Test control frequency consistency (50Hz matching simulation)
- Validate action space scaling and clipping
- Monitor real-world locomotion performance and stability