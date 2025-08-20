## Project Overview

A unified deployment framework for the Unitree Go2 quadruped robot that enables running reinforcement learning policies from various training environments. The framework provides a flexible policy interface abstraction that allows users to deploy custom policies with minimal configuration changes.

## Key Features

- **Policy Interface Abstraction**: Easy integration of policies from different training frameworks (legged_gym, IsaacLab, custom)
- **Multi-sensor Support**: RealSense depth camera integration with non-blocking capture system
- **ROS2 Integration**: Full ROS2 communication for robot control and sensor data
- **Safety Systems**: Hardware limits, emergency controls, and fail-fast error handling
- **50Hz Control Loop**: Consistent control frequency matching simulation training
- **Mode Management**: Seamless switching between sport mode, stand policy, and locomotion policy

## Architecture

### Core Components

- **`main.py`** - Main runner class (`Go2Runner`) orchestrating the control loop and system initialization
- **`go2_ros2_handler.py`** - ROS2 handler managing robot communication, sensor data processing, motor control, and observation collection
- **`depth_publisher.py`** - Non-blocking depth image capture and publishing system with clean architecture separation
- **`policy_interface/base.py`** - Abstract base class for policy interfaces
- **`policy_interface/__init__.py`** - Factory function for policy interface selection
- **`utils/control_mode_manager.py`** - State management for robot operational modes (sport/stand/locomotion)
- **`utils/config.py`** - Configuration utilities and joint mapping
- **`utils/hardware_cfgs.py`** - Hardware-specific constants and limits

### Control Flow

1. **Initialization**: `Go2Runner` creates policy interface, handler, and sport mode manager
2. **Policy Interface**: Detects and loads appropriate policy based on logdir path
3. **Configuration Loading**: Policy interface provides handler configuration (joint maps, PID gains, scaling, etc.)
4. **Depth Publisher Setup**: If vision is enabled, starts separate depth publisher process for non-blocking camera operations
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

### Supported Policies

**1. Extreme-Parkour-Onboard (EPO)**
- Training Framework: legged_gym
- Features: Advanced parkour locomotion with visual-motor coordination
- Sensors: Intel RealSense D435i depth images (87x58 resolution) + proprioceptive observations
- Status: ✅ Fully supported and tested

**2. legged-loco Base Policy**
- Training Framework: IsaacLab
- Features: Base locomotion without vision, 9-step history, 50Hz control
- Sensors: Proprioceptive observations only
- Status: ✅ Fully supported and tested

**3. ABS (Position-only Control)**
- Training Framework: Custom
- Features: Position-only control policy
- Sensors: Proprioceptive observations only
- Status: ✅ Fully supported

**Note**: legged-loco vision policy with LiDAR heightmaps is not yet supported due to heightmap data unavailability when sport mode is disabled.

### Directory Structure
```
go2-deploy/
├── main.py                 # Main runner and entry point
├── go2_ros2_handler.py     # ROS2 handler and robot control
├── depth_publisher.py      # Non-blocking depth image capture and publishing
├── policy_interface/       # Policy abstraction system
│   ├── __init__.py         # Factory function for policy selection
│   ├── base.py             # Abstract base class
│   ├── EPO.py              # Extreme-Parkour-Onboard implementation
│   ├── legged_loco.py      # IsaacLab/legged-loco implementation
│   └── ABS.py              # ABS position-only implementation
├── utils/                  # Utility modules
│   ├── __init__.py
│   ├── config.py           # Configuration utilities and joint mapping
│   ├── hardware_cfgs.py    # Hardware constants and limits
│   ├── quaternion_utils.py # Quaternion math utilities
│   └── control_mode_manager.py # Robot mode state management
├── weight-and-cfg/         # Neural network weights and configurations
│   ├── legged-loco-base/   # IsaacLab base policies
│   ├── EPO/                # Extreme-Parkour-Onboard policies
│   └── ABS/                # ABS position-only policies
├── aarch64/                # ARM64 architecture binaries
│   └── crc_module.so
├── x86/                    # x86_64 architecture binaries
│   └── crc_module.so
├── CLAUDE.md               # Project documentation
├── README.md               # Project readme
└── LICENSE                 # License file
```

## Usage Guide

### Quick Start

```bash
# Run with existing policies
python main.py --logdir weight-and-cfg/EPO --nodryrun        # Extreme-Parkour-Onboard policy
python main.py --logdir weight-and-cfg/legged-loco-base --nodryrun # legged-loco base policy
python main.py --logdir weight-and-cfg/ABS --nodryrun        # ABS position-only policy

# Debug mode (no robot movement)
python main.py --logdir <policy_path> 

```

### Implementing Custom Policies

#### 1. Create Policy Interface

Create a new file in `policy_interface/` that inherits from `BasePolicyInterface`:

```python
from policy_interface.base import BasePolicyInterface

class CustomPolicyInterface(BasePolicyInterface):
    def __init__(self, logdir, device="cpu"):
        # Load your policy weights and configs
        pass
    
    def get_configs_for_handler(self):
        # Return configuration tuple for the handler
        return (joint_map, default_joint_pos, kp, kd, action_scale, clip_obs, clip_actions)
    
    def set_handler(self, handler):
        # Store handler reference for accessing observations
        self.handler = handler
    
    def get_action(self):
        # Collect observations from handler and compute actions
        obs = self._collect_observations()
        action = self.policy(obs)
        return action
```

#### 2. Available Observations from Handler

The handler provides the following observation methods:

- **`get_xyyaw_command()`**: Command velocities (x, y, yaw)
- **`get_ang_vel_obs()`**: Angular velocities in base frame
- **`get_base_rpy_obs()`**: Roll, pitch, yaw of the base
- **`get_base_quat_obs()`**: Base orientation quaternion
- **`get_dof_pos_obs()`**: Joint positions (normalized)
- **`get_dof_vel_obs()`**: Joint velocities (normalized)
- **`get_last_actions_obs()`**: Previous action commands
- **`get_contact_filt_obs()`**: Filtered foot contact states
- **`get_depth_image()`**: Depth image tensor from RealSense camera (if enabled)
- **`get_translation()`**: Robot translation in odometry frame (if enabled)
- **`get_translation_world_frame()`**: Robot translation in world frame (if enabled)

#### 3. Provide Weight and Config Files

Create a directory structure under `weight-and-cfg/`:

```
weight-and-cfg/
└── your-policy/
    ├── policy.jit      # PyTorch JIT traced policy
    └── params/         # Optional config files
        ├── agent.yaml
        └── env.yaml
```

#### 4. Register Policy in Factory

Update `policy_interface/__init__.py` to include your policy:

```python
def get_policy_interface(logdir, device):
    # ... existing code ...
    elif "your-policy" in logdir:
        from policy_interface.custom import CustomPolicyInterface
        return CustomPolicyInterface(logdir, device)
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
- NumPy (numerical computations)
- Intel RealSense SDK (depth camera)
- Unitree SDK (robot hardware)
- PyYAML (configuration file parsing)

**Architecture Support:**
- x86_64 and aarch64 architectures
- CRC module for reliable communication

## Vision System Architecture

### Depth Image System (EPO Policies)

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

### Heightmap System (Future Enhancement)

**Note**: LiDAR-based heightmap support for legged-loco vision policies is planned but not yet implemented. The main limitation is that heightmap data from the Go2's onboard LiDAR is currently unavailable when sport mode is disabled.

### Error Handling & Safety
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

## Framework Components

### Implementation Status
- ✅ Multi-policy deployment system with flexible interface
- ✅ Non-blocking depth image capture system
- ✅ ROS2 integration and robot communication
- ✅ Sport mode management with smooth transitions
- ✅ Safety systems and hardware limit enforcement
- ✅ Support for EPO, legged-loco base, and ABS policies

### Known Limitations
- LiDAR heightmap data unavailable when sport mode is disabled (affects legged-loco vision policies)
- Depth camera required for vision-based policies (Intel RealSense D435i)

## File Structure and Development Guidelines

### Core Files
- **`main.py`**: Main runner and argument parsing
- **`go2_ros2_handler.py`**: ROS2 communication, robot control, and observation provider
- **`depth_publisher.py`**: Non-blocking depth image capture for vision policies
- **`policy_interface/base.py`**: Abstract base class for all policy interfaces
- **`policy_interface/EPO.py`**: Extreme-Parkour-Onboard policy implementation
- **`policy_interface/legged_loco.py`**: legged-loco base policy implementation
- **`policy_interface/ABS.py`**: ABS position-only policy implementation
- **`utils/`**: Utility modules (config, hardware, control mode management)

### Development Principles
- **Read-only external repositories**: Never modify training code in external repositories
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

### Adding New Observation Types

To add new observation types to the framework:

1. **Implement capture in handler or separate publisher**:
   - For blocking operations: Create separate publisher (like `depth_publisher.py`)
   - For non-blocking operations: Add directly to handler

2. **Add getter method in handler**:
   ```python
   def get_new_observation(self):
       return self.new_observation_buffer
   ```

3. **Use in policy interface**:
   ```python
   def _collect_observations(self):
       new_obs = self.handler.get_new_observation()
       # Process and return observations
   ```