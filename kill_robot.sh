#!/bin/bash
# Emergency robot process killer
# Usage: ./kill_robot.sh

echo "Killing robot processes..."

# Kill main.py processes
pkill -9 -f "python.*main.py"

# Kill any ROS2 processes
pkill -9 -f ros2

# Kill any python processes in this directory
pkill -9 -f "python.*go2"

echo "Robot processes killed."