# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a robotics deployment system for the Unitree Go2 quadruped robot designed to deploy reinforcement learning policies from multiple training environments. The system has been refactored to support policies from both legged_gym (Extreme-Parkour-Onboard) and IsaacLab (legged-loco) training environments through a policy interface abstraction.

## Current Development Goals

**Primary Objective**: Create a unified deployment system that can load and run RL policies from different training environments with minimal configuration changes.

**Current Focus**: 
- Refactoring vision-related code for improved readability and clarity
- Testing vision pipeline functionality with Extreme-Parkour-Onboard policies that use Intel RealSense D435i depth images
- Validating depth image processing and visual-motor coordination for parkour locomotion

**Future goal**: 
- Deploy legged-loco vision policies (secondary priority)

## Architecture

### Core Components

- **`main.py`** - Main runner class (`Go2Runner`) orchestrating the control loop and system initialization
- **`go2_ros2_handler.py`** - ROS2 handler managing robot communication, sensor data processing, motor control, and observation collection
- **`depth_publisher.py`** - **NEW**: Non-blocking depth image capture and publishing system with clean architecture separation
- **`policy_interface/legged_loco.py`** - IsaacLab/legged-loco policy implementation (base locomotion tested successfully)
- **`policy_interface/base.py`** - Abstract base class for policy interfaces
- **`policy_interface/__init__.py`** - Factory function for policy interface selection
- **`utils/control_mode_manager.py`** - State management for robot operational modes (sport/stand/locomotion)
- **`utils/config.py`** - Configuration utilities and joint mapping
- **`utils/hardware.py`** - Hardware-specific constants and limits

### Control Flow

1. **Initialization**: `Go2Runner` creates policy interface, handler, and sport mode manager
2. **Policy Interface**: Detects and loads appropriate policy based on logdir path
3. **Configuration Loading**: Policy interface provides handler configuration (joint maps, PID gains, scaling, etc.)
4. **Depth Publisher Setup**: **NEW** - If vision is enabled, starts separate depth publisher process for non-blocking camera operations
5. **ROS Setup**: Handler initializes ROS2 publishers, subscribers, and communication (including depth image subscription if enabled)
6. **Control Loop**: 50Hz main loop for consistent control frequency matching simulation training
7. **Mode Management**: Sport mode manager handles state transitions based on controller input:
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

**legged-loco (IsaacLab)** - Base policy successfully deployed
- Source: `~/legged-loco/logs/rsl_rl/go2_base/2025-07-03_21-32-44_XXX/`
- Directory for weight and configs loading: `weight-and-cfg/legged-loco/`
- Features: Go2 base locomotion (no vision), 9-step history, 50Hz control
- Status: Base locomotion policy successfully validated on hardware

**Extreme-Parkour-Onboard (legged_gym)** - CURRENT TESTING FOCUS
- Source: `~/Extreme-Parkour-Onboard/traced/` (vision-based policies)
- Directory for weight and configs loading: `weight-and-cfg/EPO/`
- Features: Parkour locomotion with Intel RealSense D435i depth images, visual-motor coordination
- Input: Depth images (87x58 resolution) + proprioceptive observations
- Training details: Trained in legged_gym with vision-based obstacle navigation
- Critical requirement: Vision pipeline refactoring for clarity and robust depth image processing

### Directory Structure
```
go2-deploy/
├── main.py                 # Main runner and entry point
├── go2_ros2_handler.py     # ROS2 handler and robot control
├── depth_publisher.py      # NEW: Non-blocking depth image capture and publishing
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
│   └── legged-loco/        # IsaacLab policies (base policy successfully deployed)
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
├── ~/Extreme-Parkour-Onboard/  # Vision-based parkour training repository (read-only)
│   └── traced/             # Vision policy weights for testing
```

## Development Commands

### Running the System

```bash
# Run with legged-loco policy (base policy successfully tested)
python main.py --logdir weight-and-cfg/legged-loco

# Run with Extreme-Parkour-Onboard vision policy (CURRENT TESTING FOCUS)
python main.py --logdir ~/Extreme-Parkour-Onboard/traced

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
- OpenCV (computer vision)
- NumPy (numerical computations)
- Intel RealSense SDK (depth camera)
- Unitree SDK (robot hardware)
- rsl_rl (reinforcement learning library)

**Architecture Support:**
- x86_64 and aarch64 architectures
- CRC module for reliable communication

## Depth Image Architecture

**NEW IMPLEMENTATION**: Non-blocking depth image capture system with clean architectural separation

### Components

**`depth_publisher.py`** - Three-tier architecture:
- **`DepthCaptureHandler`**: Pure RealSense camera operations in separate thread
- **`DepthImagePublisherNode`**: ROS2 node for publishing depth tensors at 100Hz
- **`DepthImagePublisherRunner`**: Orchestrates both components with lifecycle management

### Process Architecture
- **Separate Process**: Depth publisher runs as independent process via `multiprocessing.Process`
- **Non-blocking Main Loop**: 50Hz robot control never waits for camera operations
- **ROS2 Communication**: Uses `/depth_image_tensor` topic with `Float32MultiArray` messages
- **Thread Safety**: RealSense blocking calls isolated in background thread with proper locking

### Error Handling & Safety
- **Fail-Fast Design**: Camera errors cause immediate process shutdown
- **Error Signal Propagation**: Empty messages signal failure to main process
- **Assertion-Based Failure**: `get_depth_image()` uses assertions to stop main process when vision fails
- **No Silent Degradation**: System stops rather than operating with stale/missing depth data

### Configuration
- **RealSense Settings**: 640x480@30fps, depth filtering (hole filling, spatial, temporal)
- **Processing Pipeline**: Cropping, depth range clipping [0-3m], normalization, resizing
- **Output Format**: Configurable resolution depth tensors, centered around 0 ([-0.5, 0.5])

## Safety Features

- **Dryrun mode**: Testing without robot movement
- **Joint limit clipping**: Enforces hardware joint position limits
- **Torque limit clipping**: Prevents excessive motor torques
- **Contact force monitoring**: Foot contact detection for safety
- **Emergency modes**: Controller-based safety shutdown and mode switching
- **Hardware abstraction**: Safe motor control with configurable PID gains
- **Vision failure safety**: **NEW** - System stops when depth capture fails to prevent unsafe operation

## Current Development Status

### Implementation Status
- Multi-policy deployment system structure in place
- Policy interface abstraction implemented
- Configuration management system refactored
- ROS2 integration and robot communication
- **Non-blocking depth image capture system implemented** - **NEW**
- Sport mode management
- Safety systems and motor control

### Immediate Tasks
1. **Vision Code Refactoring**: Improve readability and clarity of vision processing pipeline
2. **Extreme-Parkour Policy Testing**: Deploy and test vision-based parkour policies with RealSense D435i
3. **Depth Image Processing**: Validate depth image capture, preprocessing, and integration with RL policies
4. **Visual-Motor Coordination**: Test parkour locomotion with visual obstacle detection and navigation
5. **Performance Analysis**: Monitor vision pipeline timing and policy execution performance

## File Structure and Development Guidelines

### Core Files
- **`main.py`**: Main runner and argument parsing
- **`go2_ros2_handler.py`**: ROS2 communication and robot control
- **`depth_publisher.py`**: **NEW** - Non-blocking depth image capture with clean architectural separation
- **`policy_interface/legged_loco.py`**: legged-loco policy implementation *(base policy successfully deployed)*
- **`utils/`**: Utility modules (config, hardware, sport mode management)

### Development Principles
- **Read-only external repositories**: Never modify training code in ~/legged-loco or ~/Extreme-Parkour-Onboard
- **Simulation consistency**: Maintain exact consistency between simulation training parameters and hardware deployment
- **Clean abstractions**: Maintain clear separation between policy logic, vision processing, and robot control
- **Policy interface isolation**: The handler should not be aware of specific policy interfaces - all policy-specific configuration must come through the interface's get_configs_for_handler() method
- **Safety first**: Always maintain hardware safety limits and emergency controls
- **Vision pipeline clarity**: Prioritize readable and maintainable vision processing code
- **Fail-fast over fault-tolerance**: Prioritize safety by failing fast when assumptions are violated rather than attempting to continue with potentially incorrect state. Use assertions to validate critical assumptions and stop execution when the system doesn't behave as expected.

### Configuration Management
- **Do not modify configuration files**: Configuration files in weight-and-cfg/ are copied from simulation training and should remain unchanged
- **Policy-specific configs**: Each policy interface handles its own configuration format
- **Automatic detection**: System automatically selects appropriate policy interface based on logdir path

### Vision Pipeline Refactoring Goals
The current focus is improving vision-related code and testing Extreme-Parkour policies:
- Refactor vision processing code for improved readability and maintainability  
- Test Extreme-Parkour-Onboard policies with Intel RealSense D435i depth camera
- Validate depth image capture, processing, and integration with visual-motor policies
- Ensure consistent vision pipeline between simulation training and hardware deployment
- Test parkour locomotion capabilities with visual obstacle navigation
- Monitor vision processing performance and real-time constraints (depth images at 100Hz)
- Future: Integrate legged-loco vision policies after vision pipeline stabilization