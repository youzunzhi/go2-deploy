## Project Overview

This is a robotics deployment system for the Unitree Go2 quadruped robot designed to deploy reinforcement learning policies from multiple training environments. The system has been refactored to support policies from both legged_gym (Extreme-Parkour-Onboard) and IsaacLab (legged-loco) training environments through a policy interface abstraction.

## Current Development Goals

**Primary Objective**: Create a unified deployment system that can load and run RL policies from different training environments with minimal configuration changes.

**Current Focus**: 
- Deploy legged-loco vision policies using Go2's onboard LiDAR sensor for heightmap generation
- Expand legged-loco policy interface to support vision-based observations consistent with ../legged-loco training
- Implement heightmap publisher for Go2's LiDAR sensor data processing and publishing

**Completed Goals**: 
- ✅ Vision-related code refactoring for improved readability and clarity
- ✅ Extreme-Parkour-Onboard (EPO) policy testing with Intel RealSense D435i depth images
- ✅ Depth image processing validation and visual-motor coordination for parkour locomotion

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

**legged-loco (IsaacLab)** - CURRENT FOCUS: Vision Policy Deployment
- Source: `~/legged-loco/logs/rsl_rl/go2_vision/` (vision-based policies)
- Directory for weight and configs loading: `weight-and-cfg/legged-loco-base/` (base policy), future: `weight-and-cfg/legged-loco-vision/`
- Features: 
  - ✅ Base policy: Go2 base locomotion (no vision), 9-step history, 50Hz control - **Successfully validated on hardware**
  - 🔄 Vision policy: Go2 locomotion with LiDAR-based heightmaps for terrain awareness
- Input: LiDAR heightmaps + proprioceptive observations (consistent with ../legged-loco training)
- Next steps: Expand policy interface for vision obs, implement heightmap publisher, extend ROS2 handler

**Extreme-Parkour-Onboard (legged_gym)** - COMPLETED TESTING
- Source: `~/Extreme-Parkour-Onboard/traced/` (vision-based policies)
- Directory for weight and configs loading: `weight-and-cfg/EPO/`
- Features: Parkour locomotion with Intel RealSense D435i depth images, visual-motor coordination
- Input: Depth images (87x58 resolution) + proprioceptive observations
- Status: ✅ **Successfully tested - vision pipeline refactored and EPO policy deployment validated**

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
│   ├── legged-loco-base/   # IsaacLab base policies (successfully deployed)
│   │   ├── params/
│   │   │   ├── agent.yaml
│   │   │   └── env.yaml
│   │   └── policy.jit      # Copy from ~/legged-loco base training outputs
│   ├── legged-loco-vision/ # IsaacLab vision policies (future implementation)
│   └── EPO/                # Extreme-Parkour-Onboard policies (successfully tested)
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
│   ├── logs/rsl_rl/go2_base/2025-07-03_21-32-44_XXX/exported/policy.jit  # Base policy (deployed)
│   └── logs/rsl_rl/go2_vision/                                           # Vision policy (next focus)
├── ~/Extreme-Parkour-Onboard/  # Vision-based parkour training repository (read-only)
│   └── traced/             # Vision policy weights (successfully tested)
├── ~/unitree_sdk2_python/  # Unitree SDK reference for LiDAR integration
```

## Development Commands

### Running the System

```bash
# Run with legged-loco base policy (successfully tested)
python main.py --logdir weight-and-cfg/legged-loco-base

# Run with legged-loco vision policy (CURRENT DEVELOPMENT FOCUS)
python main.py --logdir weight-and-cfg/legged-loco-vision

# Run with Extreme-Parkour-Onboard vision policy (successfully tested)
python main.py --logdir weight-and-cfg/EPO

# Debug mode without robot movement
python main.py --logdir <policy_path> --dryrun

# Specify device
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

## Vision System Architecture

### Depth Image System (EPO Policies) - ✅ COMPLETED

**Implementation**: Non-blocking depth image capture system with clean architectural separation

**`depth_publisher.py`** - Three-tier architecture:
- **`DepthCaptureHandler`**: Pure RealSense camera operations in separate thread
- **`DepthImagePublisherNode`**: ROS2 node for publishing depth tensors at 100Hz
- **`DepthImagePublisherRunner`**: Orchestrates both components with lifecycle management

**Process Architecture:**
- **Separate Process**: Depth publisher runs as independent process via `multiprocessing.Process`
- **Non-blocking Main Loop**: 50Hz robot control never waits for camera operations
- **ROS2 Communication**: Uses `/depth_image_tensor` topic with `Float32MultiArray` messages
- **Thread Safety**: RealSense blocking calls isolated in background thread with proper locking

**Configuration:**
- **RealSense Settings**: 640x480@30fps, depth filtering (hole filling, spatial, temporal)
- **Processing Pipeline**: Cropping, depth range clipping [0-3m], normalization, resizing
- **Output Format**: Configurable resolution depth tensors, centered around 0 ([-0.5, 0.5])

### Heightmap System (legged-loco Policies) - 🔄 NEXT IMPLEMENTATION

**Planned Implementation**: Non-blocking LiDAR-based heightmap capture and publishing system

**Future `heightmap_publisher.py`** - Similar architecture to depth publisher:
- **`HeightmapCaptureHandler`**: Go2 LiDAR sensor operations in separate thread
- **`HeightmapPublisherNode`**: ROS2 node for publishing heightmap tensors
- **`HeightmapPublisherRunner`**: Orchestrates both components with lifecycle management

**Process Architecture:**
- **Separate Process**: Heightmap publisher runs as independent process
- **Non-blocking Main Loop**: 50Hz robot control never waits for LiDAR operations  
- **ROS2 Communication**: Uses `/heightmap_tensor` topic with `Float32MultiArray` messages
- **SDK Integration**: Uses ~/unitree_sdk2_python for Go2 LiDAR sensor access

**Configuration:**
- **LiDAR Processing**: Point cloud to heightmap conversion, terrain analysis
- **Output Format**: Heightmap tensors consistent with ../legged-loco training format

### Error Handling & Safety (Both Systems)
- **Fail-Fast Design**: Sensor errors cause immediate process shutdown
- **Error Signal Propagation**: Empty messages signal failure to main process
- **Assertion-Based Failure**: Vision/heightmap getters use assertions to stop main process when sensors fail
- **No Silent Degradation**: System stops rather than operating with stale/missing sensor data

## Safety Features

- **Dryrun mode**: Testing without robot movement
- **Joint limit clipping**: Enforces hardware joint position limits
- **Torque limit clipping**: Prevents excessive motor torques
- **Contact force monitoring**: Foot contact detection for safety
- **Emergency modes**: Controller-based safety shutdown and mode switching
- **Hardware abstraction**: Safe motor control with configurable PID gains
- **Vision/sensor failure safety**: System stops when depth/heightmap capture fails to prevent unsafe operation

## Current Development Status

### Implementation Status
- ✅ Multi-policy deployment system structure in place
- ✅ Policy interface abstraction implemented  
- ✅ Configuration management system refactored
- ✅ ROS2 integration and robot communication
- ✅ **Non-blocking depth image capture system implemented and tested**
- ✅ **EPO vision policy deployment validated**
- ✅ Sport mode management
- ✅ Safety systems and motor control

### Current Tasks (legged-loco Vision Policy Deployment)
1. **Expand legged-loco Policy Interface**: Modify `policy_interface/legged_loco.py` to support vision-based observations consistent with ../legged-loco go2 vision policy training
2. **Implement Heightmap Publisher**: Create `heightmap_publisher.py` using Go2's onboard LiDAR sensor with architecture similar to `depth_publisher.py`, referencing ~/unitree_sdk2_python
3. **Extend ROS2 Handler**: Modify `go2_ros2_handler.py` to subscribe to heightmap data and provide it to vision policies through the policy interface
4. **Configuration Integration**: Ensure heightmap processing parameters match ../legged-loco training environment exactly
5. **Testing and Validation**: Deploy and test legged-loco vision policies on hardware

## File Structure and Development Guidelines

### Core Files
- **`main.py`**: Main runner and argument parsing
- **`go2_ros2_handler.py`**: ROS2 communication and robot control
- **`depth_publisher.py`**: ✅ Non-blocking depth image capture with clean architectural separation *(EPO policies)*
- **`heightmap_publisher.py`**: 🔄 Non-blocking LiDAR heightmap capture *(legged-loco vision policies - next implementation)*
- **`policy_interface/legged_loco.py`**: legged-loco policy implementation *(base policy successfully deployed, vision support next)*
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

### legged-loco Vision Policy Implementation Goals
The current focus is implementing legged-loco vision policy support:
- Expand legged-loco policy interface to handle vision observations (heightmaps from LiDAR)
- Implement heightmap publisher using Go2's onboard LiDAR sensor (reference ~/unitree_sdk2_python)
- Extend ROS2 handler to subscribe to heightmap data and provide it to policy interface
- Ensure heightmap processing exactly matches ../legged-loco training environment
- Deploy and validate legged-loco vision policies on hardware
- Maintain consistent sensor data format between training and deployment
)