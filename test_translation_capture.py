#!/usr/bin/env python3

"""
Test script for translation capture functionality in Go2ROS2Handler.
This script tests the translation capture feature by creating a handler with 
enable_translation_capture=True and monitoring the translation data.
"""

import rclpy
import torch
import time
import numpy as np
from go2_ros2_handler import Go2ROS2Handler

def test_translation_capture():
    """Test the translation capture functionality"""
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        # Create handler with translation capture enabled
        print("Creating Go2ROS2Handler with translation capture enabled...")
        
        # Use minimal configuration for testing
        joint_map = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # Simple 1:1 mapping
        default_joint_pos = [0.0] * 12  # Zero default positions
        
        handler = Go2ROS2Handler(
            joint_map=joint_map,
            default_joint_pos=default_joint_pos,
            kp=20.0,
            kd=0.5,
            action_scale=0.25,
            clip_obs=100.0,
            clip_actions=100.0,
            device="cpu",
            dryrun=True,
            enable_depth_capture=False,
            enable_translation_capture=True  # Enable translation capture
        )
        
        print("Handler created successfully!")
        print("Waiting for odometry data from /odometry/filtered...")
        print("Make sure the robot is publishing odometry data to /odometry/filtered")
        print("Press Ctrl+C to stop the test")
        
        # Wait for initial odometry data
        start_time = time.time()
        timeout = 10.0  # 10 second timeout
        
        while not handler.start_pos_captured and (time.time() - start_time) < timeout:
            rclpy.spin_once(handler.node, timeout_sec=0.1)
            time.sleep(0.1)
        
        if not handler.start_pos_captured:
            print("ERROR: No odometry data received within timeout period!")
            print("Please check that /odometry/filtered topic is publishing data")
            return False
        
        print(f"Start position captured: {handler.start_pos.cpu().numpy()}")
        
        # Monitor translation for a few seconds
        print("\nMonitoring translation data for 30 seconds...")
        print("Move the robot to see translation changes")
        print("Format: [x, y, z] in meters")
        
        monitor_start = time.time()
        last_print_time = 0
        
        while (time.time() - monitor_start) < 30.0:
            rclpy.spin_once(handler.node, timeout_sec=0.1)
            
            # Print translation every second
            current_time = time.time()
            if current_time - last_print_time >= 1.0:
                try:
                    translation = handler.get_translation()
                    translation_np = translation.cpu().numpy()[0]  # Get first (and only) batch element
                    print(f"Translation: [{translation_np[0]:+.3f}, {translation_np[1]:+.3f}, {translation_np[2]:+.3f}] m")
                    last_print_time = current_time
                except Exception as e:
                    print(f"Error getting translation: {e}")
            
            time.sleep(0.1)
        
        print("\nTest completed successfully!")
        return True
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        return True
    except Exception as e:
        print(f"Error during test: {e}")
        return False
    finally:
        # Clean up
        try:
            handler.node.destroy_node()
        except:
            pass
        rclpy.shutdown()

if __name__ == "__main__":
    print("=== Translation Capture Test ===")
    print("This test verifies that the Go2ROS2Handler can capture translation data")
    print("from the /odometry/filtered topic.")
    print()
    
    success = test_translation_capture()
    
    if success:
        print("\n✅ Translation capture test completed!")
    else:
        print("\n❌ Translation capture test failed!")
        exit(1)
