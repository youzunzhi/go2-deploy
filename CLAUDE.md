# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a robotics deployment system for the Unitree Go2 quadruped robot designed to deploy reinforcement learning policies from multiple training environments. The system has been refactored to support policies from both legged_gym (Extreme-Parkour-Onboard) and IsaacLab (legged-loco) training environments through a policy interface abstraction.

## Current Development Goals

**Primary Objective**: Create a unified deployment system that can load and run RL policies from different training environments with minimal configuration changes.

**Current Focus**: 
- Clean and restructure the handler file (`go2_ros2_handler.py`) for better readability and maintainability
- Code organization and optimization
- Testing and debugging the multi-policy system
- Performance optimization and timing improvements

**System Status**: 
- Policy interface abstraction implemented
- Multi-policy loading system in place
- Configuration management refactored
- Handler system needs cleanup and optimization

## Architecture

### Core Components

- **`main.py`** - Main runner class (`Go2Runner`) orchestrating the control loop and system initialization
- **`go2_ros2_handler.py`** - ROS2 handler managing robot communication, sensor data processing, motor control, and observation collection
- **`policy_interface/`** - Modular policy interface system supporting multiple training environments:
  - `base.py` - Abstract base class for policy interfaces
  - `EPO.py` - Extreme-Parkour-Onboard policy implementation
  - `legged_loco.py` - IsaacLab/legged-loco policy implementation
  - `__init__.py` - Factory function for policy interface selection
- **`utils/control_mode_manager.py`** - State management for robot operational modes (sport/stand/locomotion)
- **`utils/config.py`** - Configuration utilities and joint mapping
- **`utils/hardware.py`** - Hardware-specific constants and limits
- **`visual_node.py`** - Visual processing pipeline for Intel RealSense depth camera integration
- **`weight-and-cfg/`** - Neural network weights and configuration files organized by policy source

### Control Flow

1. **Initialization**: `Go2Runner` creates policy interface, handler, and sport mode manager
2. **Policy Interface**: Detects and loads appropriate policy based on logdir path
3. **Configuration Loading**: Policy interface provides handler configuration (joint maps, PID gains, scaling, etc.)
4. **ROS Setup**: Handler initializes ROS2 publishers, subscribers, and communication
5. **Control Loop**: 50Hz main loop with configurable timing modes:
   - **ROS Timer**: Standard ROS2-managed timing
   - **Manual Control**: Precise timing for high-performance operation
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

### Supported Policy Sources

1. **Extreme-Parkour-Onboard (EPO)**
   - Configuration: `weight-and-cfg/EPO/config.json`
   - Models: `base_jit.pt`, `vision_weight.pt`
   - Features: Depth vision integration, history encoding, state estimation

2. **legged-loco (IsaacLab)** 
   - Configuration: `weight-and-cfg/legged-loco/params/`
   - Models: `policy.jit`
   - Features: Locomotion policies

### Directory Structure
```
go2-deploy/
├── main.py                 # Main runner and entry point
├── go2_ros2_handler.py     # ROS2 handler and robot control
├── go2_controller.py       # Controller utilities
├── visual_node.py          # Visual processing pipeline
├── policy_interface/       # Policy abstraction system
│   ├── __init__.py         # Factory function for policy selection
│   ├── base.py             # Abstract base class
│   ├── EPO.py              # Extreme-Parkour-Onboard implementation
│   └── legged_loco.py      # IsaacLab implementation
├── utils/                  # Utility modules
│   ├── __init__.py
│   ├── config.py           # Configuration utilities and joint mapping
│   ├── hardware.py         # Hardware constants and limits
│   └── control_mode_manager.py # Robot mode state management
├── weight-and-cfg/         # Neural network weights and configurations
│   ├── EPO/                # Extreme-Parkour-Onboard policies
│   │   ├── config.json
│   │   ├── base_jit.pt
│   │   └── vision_weight.pt
│   ├── legged-loco/        # IsaacLab policies
│   │   ├── params/
│   │   │   ├── agent.yaml
│   │   │   └── env.yaml
│   │   └── policy.jit
│   └── [future-sources]/   # Additional training environments
├── aarch64/                # ARM64 architecture binaries
│   └── crc_module.so
├── x86/                    # x86_64 architecture binaries
│   └── crc_module.so
├── CLAUDE.md               # Project documentation for Claude Code
├── README.md               # Project readme
├── QUICKSTART.md           # Quick start guide
└── LICENSE                 # License file
```

## Development Commands

### Running the System

```bash
# Run with EPO policy
python main.py --logdir weight-and-cfg/EPO

# Run with legged-loco policy  
python main.py --logdir weight-and-cfg/legged-loco

# With specific timing mode
python main.py --timing_mode ros_timer  # or manual_control

# Debug mode without robot movement
python main.py --dryrun

# Specify device
python main.py --device cuda  # or cpu
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
1. **Handler Refactoring**: Clean up `go2_ros2_handler.py` structure and readability
2. **Testing and Debugging**: Validate multi-policy system functionality
3. **Performance Optimization**: Reduce computational overhead and improve timing
4. **Code Documentation**: Enhance inline documentation and type hints
5. **Error Handling**: Improve robustness and error recovery

## File Structure and Development Guidelines

### Core Files
- **`main.py`**: Main runner and argument parsing
- **`go2_ros2_handler.py`**: ROS2 communication and robot control *(CURRENT CLEANUP TARGET)*
- **`policy_interface/`**: Policy abstraction system
- **`utils/`**: Utility modules (config, hardware, sport mode management)

### Development Principles
- **Read-only external repositories**: Never modify legged-loco or Extreme-Parkour-Onboard code
- **Clean abstractions**: Maintain clear separation between policy logic and robot control
- **Backward compatibility**: Ensure changes don't break existing policy support
- **Safety first**: Always maintain hardware safety limits and emergency controls

### Configuration Management
- **Do not modify configuration files**: Configuration files in weight-and-cfg/ are copied from simulation training and should remain unchanged
- **Policy-specific configs**: Each policy interface handles its own configuration format
- **Automatic detection**: System automatically selects appropriate policy interface based on logdir path

### Handler Cleanup Goals
The `go2_ros2_handler.py` file is the current focus for cleanup and should be refactored for:
- Better code organization and structure
- Improved readability and maintainability  
- Clearer separation of concerns
- Enhanced documentation
- Reduced complexity in large methods
- More concise and clear code structure