import rclpy

import os.path as osp
import time
import torch
from multiprocessing import Process

from go2_ros2_handler import Go2ROS2Handler
from policy_interface import get_policy_interface
from utils.control_mode_manager import ControlModeManager
from depth_publisher import run_depth_publisher_process


class Go2Runner:
    """Runner class for Go2 robot control system"""
    
    def __init__(self, args):
        """Initialize the Go2 runner with configuration and models"""
        self.args = args
        logdir = args.logdir
        device = args.device
        self.duration = args.duration

        rclpy.init()
        
        self.policy_interface = get_policy_interface(logdir, device)
        
        # Initialize depth publisher process if needed
        self.depth_publisher_process = None

        # Get configs for handler
        joint_map, default_joint_pos, kp, kd, action_scale, clip_obs, clip_actions, enable_depth_capture, depth_resolution, enable_translation_capture = self.policy_interface.get_configs_for_handler()

        self.handler = Go2ROS2Handler(
            joint_map=joint_map,
            default_joint_pos=default_joint_pos,
            device=device,
            dryrun=not args.nodryrun,
            kp=kp,
            kd=kd,
            action_scale=action_scale,
            clip_obs=clip_obs,
            clip_actions=clip_actions,
            enable_depth_capture=enable_depth_capture,
            depth_resolution=depth_resolution,
            enable_translation_capture=enable_translation_capture,
        )

        if enable_depth_capture:
            self._start_depth_publisher_process(depth_resolution)

        # Set handler to policy interface
        self.policy_interface.set_handler(self.handler)

        self.control_mode_manager = ControlModeManager(self.handler, self.policy_interface)
        
        # Warm up policy once at startup (similar to Extreme-Parkour-Onboard)
        self._warmup_policy()
        
        # Print configuration information
        self.log_system_info()
        
    def _start_depth_publisher_process(self, depth_resolution):
        """Start depth publisher node in a separate process"""
        self.handler.log_info(f"Starting depth publisher node with resolution {depth_resolution}")
        
        # Create and start process for depth publisher node
        self.depth_publisher_process = Process(
            target=run_depth_publisher_process,
            args=(depth_resolution,),
            daemon=True
        )
        self.depth_publisher_process.start()
        
        # Give the depth publisher some time to initialize
        time.sleep(2.0)
        
        self.handler.log_info("Depth publisher node started successfully")

    def _warmup_policy(self):
        """Warm up policy at startup to avoid slow first iterations"""
        assert hasattr(self.policy_interface, 'warm_up_iter'), "Policy interface must have warm_up_iter attribute"
        for _ in range(self.policy_interface.warm_up_iter):
            _ = self.policy_interface.get_action()
        self.policy_interface.policy_iter_counter = 0
        self.handler.log_info(f"Policy warmed up with {self.policy_interface.warm_up_iter} iterations")

    def main_loop(self):
        """Main control loop for the Go2 robot - handles different operational modes based on joystick input"""
        # Check for safe exit first
        if self.handler.safe_exit_requested:
            self.handler.safe_exit()
            raise KeyboardInterrupt("Safe exit requested")
        
        self.control_mode_manager.sport_mode_before_locomotion()

        if self.control_mode_manager.which_mode == "locomotion":
            action = self.policy_interface.get_action()
            self.handler.send_action(action)

        self.control_mode_manager.sport_mode_after_locomotion()

    @torch.inference_mode()
    def run(self):
        """Run the main control loop"""
        try:
            # Start control loop
            self._handle_timing_mode()
        finally:
            # Shutdown properly
            self._cleanup()
    
    def _cleanup(self):
        """Clean up resources and shutdown nodes"""
        # Shutdown depth publisher process if it exists
        if self.depth_publisher_process and self.depth_publisher_process.is_alive():
            self.handler.log_info("Terminating depth publisher process")
            self.depth_publisher_process.terminate()
            self.depth_publisher_process.join(timeout=5.0)
            
        # Shutdown main handler and ROS
        self.handler.shutdown()
        rclpy.shutdown()
    
    def _start_main_loop_timer(self):
        """Start the main loop timer for ROS-based timing control"""
        self.handler.node.create_timer(
            self.duration, # in sec
            self.main_loop,
        )
    
    def _handle_timing_mode(self):
        """Handle different timing modes for the control loop"""
        if self.args.timing_mode == "ros_timer":
            # Use ROS timer for timing control
            self.handler.log_info('Model and Policy are ready')
            self._start_main_loop_timer()
            rclpy.spin(self.handler.node)
        
        elif self.args.timing_mode == "manual_control":
            # Manually control timing for more precise control
            rclpy.spin_once(self.handler.node, timeout_sec=0.)
            self.handler.log_info("Model and Policy are ready")
            
            while rclpy.ok():
                # Track iteration time to maintain desired frequency
                main_loop_time = time.monotonic()
                
                # Run one iteration
                self.main_loop()
                rclpy.spin_once(self.handler.node, timeout_sec=0.)
            
                # Sleep remaining time to maintain frequency
                sleep_time = max(0, self.duration - (time.monotonic() - main_loop_time))
                time.sleep(sleep_time)
        
        else:
            raise ValueError(f"Invalid timing mode: {self.args.timing_mode}")
    
    def log_system_info(self):
        """
        Print system configuration information
        
        Args:
            handler: Go2 Handler
            logdir: Model directory
        """
        self.handler.log_info("Model loaded from: {}".format(osp.join(self.args.logdir)))
        self.handler.log_info("Motor Stiffness (kp): {}".format(self.handler.kp))
        self.handler.log_info("Motor Damping (kd): {}".format(self.handler.kd))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--logdir", type=str, default='weight-and-cfg/ABS', help="The directory which contains the config and model weights files")
    parser.add_argument("--nodryrun", action="store_true", default=False, help="Disable dryrun mode")
    parser.add_argument("--timing_mode", type=str, default="ros_timer",
        choices=["manual_control", "ros_timer"],
        help="Select timing mode: manual_control (precise timing control) or ros_timer (ROS managed timer)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for the model")
    parser.add_argument("--duration", type=float, default=0.02, help="Control cycle duration")
    args = parser.parse_args()
    
    Go2Runner(args).run()
