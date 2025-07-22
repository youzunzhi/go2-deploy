# Go2 Deployment System

A unified deployment system for reinforcement learning policies on the Unitree Go2 quadruped robot. This codebase is adapted from [Extreme-Parkour-Onboard](https://github.com/change-every/Extreme-Parkour-Onboard).

## Project Goal

Create a unified deployment system that can load and run RL policies from different training environments (legged_gym, IsaacLab) with minimal configuration changes. The system provides a clean policy interface abstraction to handle the complexity of different training frameworks while maintaining consistent hardware deployment.

## Current Status

**✅ Successfully Completed:**
- legged-loco base locomotion policy deployment and validation

**🚧 Currently in Development:**
- Vision pipeline refactoring for improved readability and clarity
- Testing vision-based policies from Extreme-Parkour-Onboard
- Deploy legged-loco vision policies
- Extended multi-environment policy support

## Development Status

This repository is actively under development and testing. Usage manuals and detailed documentation will be provided after the development and testing phase is completed.

## Acknowledgments

This codebase is adapted from the excellent work at [Extreme-Parkour-Onboard](https://github.com/change-every/Extreme-Parkour-Onboard). Special thanks to the original authors for their contributions to legged robot deployment.

## Safety Notice

This system controls a physical robot. Always ensure proper safety measures are in place during development and testing.