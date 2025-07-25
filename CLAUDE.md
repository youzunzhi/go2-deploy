# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a robotics deployment system for the Unitree Go2 quadruped robot designed to deploy reinforcement learning policies from multiple training environments. The system has been refactored to support policies from both legged_gym (Extreme-Parkour-Onboard) and IsaacLab (legged-loco) training environments through a policy interface abstraction.

## Current Development Goals

**Primary Objective**: Unified deployment system supporting both base and vision policies from different training environments.

**COMPLETED**: 
- ✅ Base policy deployment (legged-loco) tested successfully
- ✅ LiDAR height map integration for vision policies
- ✅ Automatic policy detection (base vs vision)
- ✅ Unitree L1 LiDAR support through SDK2

**CURRENT STATUS**: 
- 🎯 **Ready for vision policy testing** - legged-loco vision policies with height map support
- 🔧 LiDAR-based height map processing pipeline implemented
- 📊 Vision policy interface with 909-dimensional observation space (45 current + 405 history + 459 height map)

**NEXT STEPS**: 
- Test vision policy deployment with vision_policy.jit
- Validate height map data flow and dimensions
- Performance optimization and real-time constraints validation

## Architecture

### Core Components

- **`main.py`** - Main runner class (`Go2Runner`) orchestrating the control loop and system initialization
- **`go2_ros2_handler.py`** - ROS2 handler managing robot communication, sensor data processing, motor control, and LiDAR height map integration
- **`policy_interface/legged_loco.py`** - IsaacLab/legged-loco policy implementations:
  - `LeggedLocoPolicyInterface` - Base locomotion (tested successfully)
  - `LeggedLocoVisionPolicyInterface` - Vision-based navigation with height map support
- **`lidar_height_map_processor.py`** - LiDAR height map processing following legged-loco paper algorithm
- **`policy_interface/base.py`** - Abstract base class for policy interfaces
- **`policy_interface/__init__.py`** - Automatic policy detection and factory function
- **`policy_interface/EPO.py`** - Extreme-Parkour-Onboard policy implementation with depth camera support
- **`utils/control_mode_manager.py`** - State management for robot operational modes (sport/stand/locomotion)
- **`utils/config.py`** - Configuration utilities and joint mapping
- **`utils/hardware_cfgs.py`** - Hardware-specific constants, limits, and LiDAR topic configurations

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

### Policy Sources

**legged-loco (IsaacLab)** - Base and Vision policies supported
- Source: `~/legged-loco/logs/rsl_rl/go2_base/2025-07-03_21-32-44_XXX/`
- Directory for weight and configs loading: `weight-and-cfg/legged-loco/`
- Features: 
  - **Base Policy** (`policy.jit`): Go2 base locomotion, 450-dim obs (45 current + 405 history), 50Hz control ✅ VALIDATED
  - **Vision Policy** (`vision_policy.jit`): Terrain-aware navigation with LiDAR height map, 909-dim obs (45 current + 405 history + 459 height map)
- LiDAR Integration: Unitree L1 LiDAR with height map processing following paper algorithm
- Automatic Detection: System automatically selects base vs vision policy based on available .jit files

**Extreme-Parkour-Onboard (legged_gym)** - Depth camera based vision
- Source: `~/Extreme-Parkour-Onboard/traced/` (vision-based policies)
- Directory for weight and configs loading: `weight-and-cfg/EPO/`
- Features: Parkour locomotion with Intel RealSense D435i depth images, visual-motor coordination
- Input: Depth images (87x58 resolution) + proprioceptive observations
- Training details: Trained in legged_gym with vision-based obstacle navigation
- Status: Separate vision pipeline using depth camera (different from legged-loco's LiDAR approach)

### Directory Structure
```
go2-deploy/
├── main.py                 # Main runner and entry point
├── go2_ros2_handler.py     # ROS2 handler and robot control
├── go2_controller.py       # Controller utilities
├── policy_interface/       # Policy abstraction system
│   ├── __init__.py         # Automatic policy detection and factory function
│   ├── base.py             # Abstract base class
│   ├── legged_loco.py      # IsaacLab implementations (base + vision)
│   └── EPO.py              # Extreme-Parkour-Onboard implementation
├── lidar_height_map_processor.py  # LiDAR height map processing for vision policies
├── utils/                  # Utility modules
│   ├── __init__.py
│   ├── config.py           # Configuration utilities and joint mapping
│   ├── hardware.py         # Hardware constants and limits
│   └── control_mode_manager.py # Robot mode state management
├── weight-and-cfg/         # Neural network weights and configurations
│   ├── legged-loco/        # IsaacLab policies
│   │   ├── params/
│   │   │   ├── agent.yaml
│   │   │   └── env.yaml
│   │   ├── policy.jit      # Base policy (validated ✅)
│   │   └── vision_policy.jit # Vision policy with height map support
│   └── EPO/                # Extreme-Parkour-Onboard policies
│       ├── base_jit.pt
│       ├── config.json
│       └── vision_weight.pt
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
├── ~/Extreme-Parkour-Onboard/  # Vision-based parkour training repository (read-only)
│   └── traced/             # Vision policy weights for testing
```

## Development Commands

### Running the System

```bash
# Run with legged-loco base policy (validated ✅)
python main.py --logdir weight-and-cfg/legged-loco
# → Automatically detects policy.jit and loads LeggedLocoPolicyInterface

# Run with legged-loco vision policy (with height map support 🎯)
python main.py --logdir weight-and-cfg/legged-loco
# → Automatically detects vision_policy.jit and loads LeggedLocoVisionPolicyInterface
# → Enables LiDAR height map processing pipeline

# Run with Extreme-Parkour-Onboard vision policy (depth camera based)
python main.py --logdir ~/Extreme-Parkour-Onboard/traced
# → Loads EPOPolicyInterface with depth camera processing

# Debug mode without robot movement
python main.py --logdir <policy_path> --dryrun

# Specify device
python main.py --logdir <policy_path> --device cuda  # or cpu
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
- NumPy (numerical computations)
- Intel RealSense SDK (depth camera for EPO policies)
- Unitree SDK2 (robot hardware and LiDAR integration)
- Unitree L1 LiDAR (height map generation for vision policies)
- rsl_rl (reinforcement learning library)
- SciPy (height map processing and filtering)

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

### Implementation Status ✅ COMPLETED
- Multi-policy deployment system structure in place
- Policy interface abstraction implemented and extended
- Configuration management system refactored  
- ROS2 integration and robot communication
- **LiDAR height map processing pipeline** - NEW
- **Automatic policy detection (base vs vision)** - NEW
- **Unitree SDK2 integration for LiDAR data** - NEW
- Sport mode management
- Safety systems and motor control

### Current Capabilities
1. **Base Policy Deployment** ✅ - legged-loco base policy validated on hardware
2. **Vision Policy Support** 🎯 - LiDAR height map integration ready for testing
3. **Height Map Processing** - Following legged-loco paper algorithm (17×27 grid, 459 dims)
4. **Dual Vision Systems** - LiDAR-based (legged-loco) and depth camera-based (EPO)
5. **Automatic Detection** - System selects appropriate interface based on available model files

### Next Testing Phase
1. **Vision Policy Validation**: Test vision_policy.jit with LiDAR height map data
2. **Observation Dimension Verification**: Ensure 909-dim obs matches model expectations
3. **LiDAR Data Flow Testing**: Validate Unitree L1 LiDAR integration and processing
4. **Performance Optimization**: Monitor real-time constraints and processing efficiency
5. **Terrain Navigation**: Test vision-guided locomotion with height map awareness

## File Structure and Development Guidelines

### Core Files
- **`main.py`**: Main runner and argument parsing
- **`go2_ros2_handler.py`**: ROS2 communication, robot control, and LiDAR integration
- **`policy_interface/legged_loco.py`**: 
  - `LeggedLocoPolicyInterface` - Base policy ✅ validated
  - `LeggedLocoVisionPolicyInterface` - Vision policy with height map support 🎯
- **`lidar_height_map_processor.py`**: LiDAR height map generation following paper algorithm
- **`policy_interface/EPO.py`**: EPO policy with depth camera processing
- **`utils/`**: Utility modules (config, hardware, sport mode management)

### Development Principles
- **Read-only external repositories**: Never modify training code in ~/legged-loco or ~/Extreme-Parkour-Onboard
- **Simulation consistency**: Maintain exact consistency between simulation training parameters and hardware deployment
- **Clean abstractions**: Maintain clear separation between policy logic, vision processing, and robot control
- **Safety first**: Always maintain hardware safety limits and emergency controls
- **LiDAR integration**: Follow legged-loco paper algorithm for height map generation
- **Sensor flexibility**: Support both LiDAR-based and depth camera-based vision systems

### Configuration Management
- **Do not modify configuration files**: Configuration files in weight-and-cfg/ are copied from simulation training and should remain unchanged
- **Policy-specific configs**: Each policy interface handles its own configuration format
- **Automatic detection**: System automatically selects appropriate policy interface based on logdir path

### Vision System Architecture
The system now supports dual vision approaches for different policy types:

**LiDAR-based Vision (legged-loco)**:
- Unitree L1 LiDAR sensor with 360° × 90° FOV
- Height map generation following paper algorithm (17×27 grid, 6cm resolution)
- 2.5D terrain representation with maximum filtering over 5 frames
- Height map dimensions: 459 elements (17×27 flattened)
- Total observation space: 909 dimensions (45 current + 405 history + 459 height map)

**Depth Camera-based Vision (EPO)**:
- Intel RealSense D435i depth camera
- Depth image processing (87×58 resolution)
- CNN-based depth encoding for parkour navigation
- Separate processing pipeline optimized for obstacle avoidance

**Key Implementation Details**:
- Automatic sensor selection based on policy type
- Unified observation interface despite different sensor modalities  
- Real-time processing constraints maintained for 50Hz control loop
- Paper-accurate algorithms ensure consistency with training environments