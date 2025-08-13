#!/usr/bin/env python3
"""
Test script to demonstrate goal setting functionality for ABS policy interface.
This shows when and how set_goal() is called in the deployment system.
"""

import torch
from policy_interface import get_policy_interface
from utils.hardware_cfgs import WirelessButtons

def test_goal_setting():
    """Test goal setting functionality"""
    
    # Simulate creating an ABS policy interface
    print("=== Testing ABS Policy Interface Goal Setting ===\n")
    
    # This would normally be called with an actual ABS model directory
    # policy_interface = get_policy_interface("path/to/ABS/model", "cpu")
    
    # For demonstration, let's show the flow:
    print("1. When locomotion mode is activated:")
    print("   - switch_to_locomotion_policy() is called")
    print("   - If policy.supports_goal_setting() returns True:")
    print("   - policy.set_goal(5.0, 0.0, 0.0) is called automatically")
    print("   - Initial goal: Forward 5 meters")
    print()
    
    print("2. During locomotion, user can press buttons to set new goals:")
    print("   - ↑ (up):    set_goal(5.0, 0.0, 0.0)     # Forward 5m")
    print("   - ↓ (down):  set_goal(-3.0, 0.0, 3.14)   # Backward 3m, turn around")
    print("   - ← (left):  set_goal(3.0, 3.0, 1.57)    # Left 3m, turn left")
    print("   - → (right): set_goal(3.0, -3.0, -1.57)  # Right 3m, turn right")
    print("   - A button:  set_goal(0.0, 0.0, 0.0)     # Return to origin")
    print()
    
    print("3. What happens when set_goal() is called:")
    print("   - Updates self.goal_pose with new target position")
    print("   - Computes heading_target = goal_heading + atan2(dy, dx)")
    print("   - This heading_target remains FIXED until next goal is set")
    print("   - During each policy step:")
    print("     * Position commands update based on current robot position")
    print("     * Heading command uses the FIXED heading_target")
    print()
    
    print("4. This matches the training behavior:")
    print("   - In training: heading_targets computed once during resampling")
    print("   - In deployment: heading_target computed once when goal is set")
    print("   - Both use: heading_target = base_heading + atan2(dy, dx)")
    print()
    
    # Show button mappings
    print("Button mappings:")
    button_mappings = {
        "up": WirelessButtons.up,
        "down": WirelessButtons.down, 
        "left": WirelessButtons.left,
        "right": WirelessButtons.right,
        "A": WirelessButtons.A,
        "L2": WirelessButtons.L2
    }
    
    for name, value in button_mappings.items():
        print(f"   {name:>5}: 0b{value:016b} ({value})")

if __name__ == "__main__":
    test_goal_setting()
