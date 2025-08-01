# Go2 Deployment System

A unified deployment system for reinforcement learning policies on the Unitree Go2 quadruped robot. 

## Project Goal

Create a unified deployment system that can load and run RL policies from different training environments (legged_gym, IsaacLab) with minimal configuration changes. The system provides a clean policy interface abstraction to handle the complexity of different training frameworks while maintaining consistent hardware deployment.

## Current Status

**✅ Successfully Completed:**
- legged-loco base locomotion policy deployment and validation
- Extreme-Parkour-Onboard (EPO) vision policy deployment and validation

**🚧 Currently in Development:**
- Deploy legged-loco vision policies using Go2's onboard LiDAR sensor
- Extended multi-environment policy support

## Development Status

This repository is actively under development and testing. Usage manuals and detailed documentation will be provided after the development and testing phase is completed.

## Acknowledgments

This codebase is adapted from the excellent work at [Extreme-Parkour-Onboard](https://github.com/change-every/Extreme-Parkour-Onboard). Special thanks to the original authors for their contributions to legged robot deployment.

## Controller Input

### Safety Controls
- **SELECT**: Emergency safe exit - immediately turns off all motors and exits the program

### Operational Controls
- **L1**: Switch from sport mode to stand policy
- **Y**: Switch from stand policy to locomotion policy  
- **L2**: Switch back to sport mode from any mode
- **R1**: Stand up (in sport mode)
- **R2**: Sit down (in sport mode)
- **X**: Balance stand (in sport mode)

## Safety Notice

This system controls a physical robot. Always ensure proper safety measures are in place during development and testing. Use the SELECT button for emergency shutdown.