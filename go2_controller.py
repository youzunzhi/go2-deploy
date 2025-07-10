#!/usr/bin/env python3
import torch  # Pre-import to resolve TLS issue
"""
Go2 Velocity Command Controller
Receives velocity commands from NaVILA server and prints them for locomotion policy
"""

import io
import json
import time
import logging
import os
from typing import Dict, Optional, Any
from threading import Lock

import cv2
import numpy as np
import requests
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import pyrealsense2 as rs
from unitree_ros2_real import UnitreeRos2Real

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Go2VelocityController(UnitreeRos2Real):
    """
    ROS2 node that receives velocity commands from NaVILA server
    and controls the robot using the Unitree Sport API.
    """
    
    def __init__(self, server_ip: str, server_port: int, instruction: str, dryrun: bool = True):
        # Initialize the parent class (UnitreeRos2Real)
        super().__init__(
            node_name='go2_velocity_controller',
            dryrun=dryrun
        )
        
        self._setup_ros_interfaces()

        # Server configuration
        self.server_url = f"http://{server_ip}:{server_port}"
        self.instruction = instruction
        
        # Image logging setup
        self.image_log_dir = "/home/unitree/go2-deploy/captured_images"
        self.output_image_path = os.path.join(self.image_log_dir, "output_img.jpg")
        os.makedirs(self.image_log_dir, exist_ok=True)
        # logger.info(f"📷 Saving captured images to '{self.output_image_path}'")

        # Camera and processing setup
        self.bridge = CvBridge()
        self._setup_camera()
        
        # Robot state
        self.is_processing = False
        self.processing_lock = Lock()
        self.use_sport_mode = True  # Start in sport mode for safety
        self.use_vlm_mode = False
        
        # VLM command execution state
        self.vlm_state = "IDLE"  # "IDLE" or "EXECUTING"
        self.vlm_command_end_time = 0.0
        
        # Velocity command tracking
        self.current_velocity = {"linear_x": 0.0, "angular_z": 0.0, "duration": 0.0}
        self.last_velocity_time = 0.0
        
        # Statistics
        self.stats = {
            'total_commands': 0,
            'successful_inferences': 0,
            'failed_inferences': 0,
            'last_inference_time': 0.0,
            'last_image_time': 0.0
        }
        
        # Initialize connection to server
        self._initialize_server_connection()
        
        # The main loop is now driven by a timer for periodic execution
        self.control_timer = self.create_timer(0.1, self.main_loop) # Run at 10Hz
        
        logger.info(f"🤖 Go2 Velocity Controller initialized")
        logger.info(f"📡 Server: {self.server_url}")
        logger.info(f"🎯 Instruction: '{self.instruction}'")
        logger.info("Press 'A' on the joystick to enter VLM mode.")
        logger.info("Press 'L2' on the joystick to exit VLM mode and return to sport mode.")

    def _setup_camera(self):
        """Initialize the RealSense camera using the verified stable method."""
        logger.info("📷 Initializing RealSense camera...")
        try:
            self.rs_pipeline = rs.pipeline()
            self.rs_config = rs.config()

            # Configure and enable only the color stream, which was proven to be stable.
            # Using 1280x720 for a wider field of view and better quality.
            width, height, fps = 1280, 720, 30
            logger.info(f"Configuring stream: Color {width}x{height} @ {fps}fps...")
            self.rs_config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            
            logger.info("Starting camera pipeline...")
            self.rs_profile = self.rs_pipeline.start(self.rs_config)
            
            # Warm up the camera to allow auto-exposure to settle.
            logger.info("📷 Warming up the camera for auto-exposure...")
            for _ in range(30):
                self.rs_pipeline.wait_for_frames()

            logger.info("✅ RealSense camera initialized and ready.")
        except Exception as e:
            logger.error(f"❌ Failed to initialize RealSense camera: {e}")
            raise RuntimeError(f"Could not initialize RealSense camera: {e}")
    
    def _setup_ros_interfaces(self):
        """Setup ROS2 publishers for monitoring and debugging."""
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.velocity_cmd_pub = self.create_publisher(
            String, '/navila/velocity_command', qos_profile
        )
        self.status_pub = self.create_publisher(
            String, '/navila/status', qos_profile
        )
        logger.info("📡 ROS2 publishers for monitoring are initialized.")
    
    def _get_image_from_camera(self) -> Optional[np.ndarray]:
        """Gets a color image frame from the RealSense camera."""
        try:
            frames = self.rs_pipeline.wait_for_frames(1000)  # Increased timeout to 1000ms
            color_frame = frames.get_color_frame()
            if not color_frame:
                logger.warning("📷 No color frame received from camera")
                return None

            color_image = np.asanyarray(color_frame.get_data())
            self.stats['last_image_time'] = time.time()
            return color_image
        except Exception as e:
            logger.error(f"❌ Failed to get image from camera: {e}")
            return None
    
    def _initialize_server_connection(self):
        """Initialize connection to VLM velocity command server"""
        max_retries = 5
        retry_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                logger.info(f"🔌 Connecting to VLM server (attempt {attempt + 1}/{max_retries})")
                
                # Test server health
                response = requests.get(f"{self.server_url}/", timeout=5)
                if response.status_code == 200:
                    server_info = response.json()
                    logger.info(f"✅ Server healthy: {server_info.get('server_type', 'unknown')}")
                    
                    # Reset agent with instruction
                    reset_response = requests.post(
                        f"{self.server_url}/reset",
                        data={'instruction': self.instruction},
                        timeout=10
                    )
                    
                    if reset_response.status_code == 200:
                        logger.info("✅ Agent reset successful")
                        return
                    else:
                        logger.error(f"❌ Agent reset failed: {reset_response.text}")
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"⚠️ Connection attempt failed: {e}")
                
                if attempt < max_retries - 1:
                    logger.info(f"⏳ Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 1.5
        
        logger.error("❌ Failed to connect to server after all retries")
        raise RuntimeError("Could not establish server connection")
    
    def _send_image_to_server(self, cv_image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Send image to VLM server and get velocity command"""
        try:
            # Convert ROS Image to OpenCV format is no longer needed
            
            # Encode as JPEG for transmission
            is_success, buffer = cv2.imencode('.jpg', cv_image, [
                cv2.IMWRITE_JPEG_QUALITY, 85
            ])
            
            if not is_success:
                logger.error("❌ Failed to encode image")
                return None
            
            # Prepare request
            files = {
                'file': ('image.jpg', io.BytesIO(buffer.tobytes()), 'image/jpeg')
            }
            
            # Send request with timeout
            start_time = time.time()
            response = requests.post(
                f"{self.server_url}/get_velocity_command",  # Updated endpoint
                files=files,
                timeout=20  # VLM inference can be slow
            )
            
            inference_time = time.time() - start_time
            self.stats['last_inference_time'] = inference_time
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success', False):
                    self.stats['successful_inferences'] += 1
                    # logger.info(f"✅ Velocity command received in {inference_time:.2f}s")
                    return result
                else:
                    logger.error(f"❌ Server error: {result.get('message', 'Unknown error')}")
            else:
                logger.error(f"❌ HTTP error {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            logger.error("⏰ Request timeout - VLM server may be overloaded")
        except requests.exceptions.RequestException as e:
            logger.error(f"🌐 Network error: {e}")
        except Exception as e:
            logger.error(f"💥 Unexpected error: {e}")
        
        self.stats['failed_inferences'] += 1
        return None
    
    def _process_and_execute_command(self, velocity_result: Dict[str, Any]):
        """Process velocity command and control the robot via Sport API."""
        
        # Print the raw VLM output as requested
        if velocity_result:
            logger.info(f"VLM action: {velocity_result.get('action_type', '')}")

        
        linear_x = velocity_result.get('linear_x', 0.0)
        angular_z = velocity_result.get('angular_z', 0.0)
        duration = velocity_result.get('duration', 0.8) # Duration is informational now
        action_type = velocity_result.get('action_type', 'unknown')
        
        self.current_velocity = {
            "linear_x": linear_x, "angular_z": angular_z, "duration": duration
        }
        self.last_velocity_time = time.time()
        
        # logger.info(
        #     f"✅ VLM command received: action='{action_type}', "
        #     f"vx={linear_x:.2f} m/s, vyaw={angular_z:.2f} rad/s"
        # )
        
        # Execute move command directly
        self.move(vx=linear_x, vy=0.0, vyaw=angular_z)

        # Publish for monitoring
        self.stats['total_commands'] += 1
        status_msg = String(data=json.dumps(self.get_status_summary()))
        self.status_pub.publish(status_msg)
        self.velocity_cmd_pub.publish(String(data=json.dumps(self.current_velocity)))

    def main_loop(self):
        """
        Main control loop, implements a state machine based on joystick input.
        """
        # Check joystick buffer safely. If it's the initial empty dict, or not yet populated, do nothing.
        if not self.joy_stick_buffer or isinstance(self.joy_stick_buffer, dict):
            return

        keys = self.joy_stick_buffer.keys
        
        # L2 is the universal "exit to sport mode" button
        if keys & self.WirelessButtons.L2:
            if self.use_vlm_mode:
                logger.info("L2 pressed. Exiting VLM mode, returning to safety standby.")
                self.use_vlm_mode = False
                self.use_sport_mode = True
                self.balance_stand() # Go to a safe, stable state
            return

        # --- Sport Mode ---
        if self.use_sport_mode:
            # Handle standard sport mode commands if needed (e.g., stand up/down)
            if keys & self.WirelessButtons.R1:
                logger.info("R1 pressed in Sport Mode: Standing up.")
                self.stand()
            if keys & self.WirelessButtons.R2:
                logger.info("R2 pressed in Sport Mode: Sitting down.")
                self.sit_down()

            # 'A' button to enter VLM mode
            if keys & self.WirelessButtons.A:
                logger.info("'A' pressed. Staring VLM mode activation sequence...")
                self.use_sport_mode = False
                self.use_vlm_mode = True
                
                # --- Activation Sequence ---
                # 1. Command the robot to stand up.
                logger.info("Activation Step 1: Commanding robot to stand up...")
                self.stand()
                time.sleep(2.0) # Give it time to complete the stand action
                
                # 2. Command the robot to enter balance standby mode. This "unlocks" it for movement.
                logger.info("Activation Step 2: Commanding robot to enter balance standby...")
                self.balance_stand()
                time.sleep(1.0) # Settle into balance stand
                
                logger.info("✅ Robot is now in VLM mode and ready to receive move commands.")
            return

        # --- VLM Mode ---
        if self.use_vlm_mode:
            now = time.time()

            # State 1: Command is actively being executed.
            if self.vlm_state == "EXECUTING":
                if now < self.vlm_command_end_time:
                    # Robot continues with the last sent velocity. Do nothing.
                    return
                else:
                    # Duration is over. Stop the robot and switch to IDLE to get the next command.
                    self.move(vx=0.0, vy=0.0, vyaw=0.0)
                    self.vlm_state = "IDLE"

            # State 2: Robot is idle, ready for a new command.
            if self.vlm_state == "IDLE":
                # --- This block replaces the previous continuous request logic ---
                if self.is_processing:
                    return

                with self.processing_lock:
                    self.is_processing = True

                try:
                    # logger.info("Requesting new command from VLM server...") # Canceled as per request
                    cv_image = self._get_image_from_camera()
                    if cv_image is not None:
                        # Save the captured image to the specified file
                        try:
                            cv2.imwrite(self.output_image_path, cv_image)
                        except Exception as e:
                            logger.error(f"❌ Failed to save image to {self.output_image_path}: {e}")

                        velocity_result = self._send_image_to_server(cv_image)
                        if velocity_result:
                            # Command received, process and execute it
                            self._process_and_execute_command(velocity_result)
                            
                            # Set the execution timer and switch state
                            duration = velocity_result.get('duration', 0.8)
                            self.vlm_command_end_time = time.time() + duration
                            self.vlm_state = "EXECUTING"
                            # logger.info(f"Command execution started. Duration: {duration:.2f}s.") # This can be noisy
                        else:
                            # If VLM fails, do nothing and wait for the next cycle to retry
                            logger.warning("VLM inference failed. Will retry on next cycle.")
                finally:
                    with self.processing_lock:
                        self.is_processing = False

    def get_status_summary(self) -> Dict[str, Any]:
        """Gets a summary of the current node status for logging."""
        current_time = time.time()
        
        last_image_time = self.stats.get('last_image_time', 0)
        image_age = current_time - last_image_time if last_image_time > 0 else -1
        
        return {
            'node_status': 'active',
            'server_url': self.server_url,
            'instruction': self.instruction,
            'processing': self.is_processing,
            'has_image': last_image_time > 0,
            'image_age': image_age,
            'current_velocity': self.current_velocity,
            'last_velocity_time': self.last_velocity_time,
            'stats': self.stats.copy()
        }
    
    def shutdown(self):
        """Gracefully shutdown the node and camera."""
        logger.info("Shutting down Go2 Velocity Controller...")
        self.rs_pipeline.stop()
        logger.info("✅ RealSense camera stopped.")
        self.destroy_node()

def main(args=None):
    rclpy.init(args=args)
    # --- Configuration ---
    SERVER_IP = os.environ.get("NAVILA_SERVER_IP", "10.165.232.223")
    SERVER_PORT = int(os.environ.get("NAVILA_SERVER_PORT", 8888))
    INSTRUCTION = "Move forward.Turn left at the navy armchair."
    DRYRUN = False # Set to False to send commands to the real robot
    
    controller_node = None
    try:
        controller_node = Go2VelocityController(
            server_ip=SERVER_IP, 
            server_port=SERVER_PORT, 
            instruction=INSTRUCTION,
            dryrun=DRYRUN
        )
        rclpy.spin(controller_node)
    except (RuntimeError, KeyboardInterrupt) as e:
        logger.info(f"Node shutdown requested: {e}")
    finally:
        if controller_node:
            controller_node.shutdown()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()