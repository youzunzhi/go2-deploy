#!/usr/bin/env python3
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Go2VelocityController(Node):
    """
    ROS2 node that receives velocity commands from NaVILA server
    Prints velocity commands for integration with locomotion policy
    """
    
    def __init__(self, server_ip: str, server_port: int, instruction: str):
        super().__init__('go2_velocity_controller')
        
        # Server configuration
        self.server_url = f"http://{server_ip}:{server_port}"
        self.instruction = instruction
        
        # Image logging setup
        self.image_log_dir = "captured_images"
        os.makedirs(self.image_log_dir, exist_ok=True)
        logger.info(f"📷 Saving captured images to '{self.image_log_dir}/'")

        # ROS2 and Camera setup
        self.bridge = CvBridge()
        self._setup_ros_interfaces()
        self._setup_camera()
        
        # Robot state
        self.is_processing = False
        self.processing_lock = Lock()
        
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
        
        # Create control timer - runs at 1Hz for VLM inference (VLM is slower than locomotion)
        self.control_timer = self.create_timer(1.0, self.control_loop)
        
        logger.info(f"🤖 Go2 Velocity Controller initialized")
        logger.info(f"📡 Server: {self.server_url}")
        logger.info(f"🎯 Instruction: '{self.instruction}'")
    
    def _setup_camera(self):
        """Initialize the RealSense camera using the verified stable method."""
        logger.info("📷 Initializing RealSense camera...")
        try:
            self.rs_pipeline = rs.pipeline()
            self.rs_config = rs.config()

            # Configure and enable only the color stream, which was proven to be stable.
            width, height, fps = 640, 480, 30
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
        """Setup ROS2 publishers"""
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Publishers
        self.velocity_cmd_pub = self.create_publisher(
            String, '/navila/velocity_command', qos_profile
        )
        self.status_pub = self.create_publisher(
            String, '/navila/status', qos_profile
        )
        
        # Image subscribers are no longer needed, camera is accessed directly
        logger.info("📡 ROS2 publishers initialized")
    
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
                    logger.info(f"✅ Velocity command received in {inference_time:.2f}s")
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
    
    def _process_velocity_command(self, velocity_result: Dict[str, Any]):
        """Process and print velocity command for locomotion policy integration"""
        
        # Extract velocity command
        linear_x = velocity_result.get('linear_x', 0.0)
        angular_z = velocity_result.get('angular_z', 0.0)
        duration = velocity_result.get('duration', 0.8)
        action_type = velocity_result.get('action_type', 'unknown')
        
        # Update current velocity state
        self.current_velocity = {
            "linear_x": linear_x,
            "angular_z": angular_z,
            "duration": duration
        }
        self.last_velocity_time = time.time()
        
        # Print velocity command for debugging and integration
        print("=" * 60)
        print(f"🎯 VELOCITY COMMAND FROM NAVILA VLM:")
        print(f"   Linear X:  {linear_x:.4f} m/s")
        print(f"   Angular Z: {angular_z:.4f} rad/s")
        print(f"   Duration:  {duration:.2f} seconds")
        print(f"   Action:    {action_type}")
        print(f"   From Queue: {velocity_result.get('from_queue', False)}")
        print(f"   Queue Remaining: {velocity_result.get('queue_remaining', 0)}")
        print(f"   Episode Step: {velocity_result.get('episode_step', 0)}")
        print(f"   Inference Time: {velocity_result.get('inference_time', 0):.3f}s")
        if 'raw_output' in velocity_result:
            print(f"   VLM Output: '{velocity_result['raw_output']}'")
        print("=" * 60)
        
        # Publish velocity command as ROS message for other nodes
        velocity_msg = String()
        velocity_msg.data = json.dumps({
            'linear_x': linear_x,
            'angular_z': angular_z,
            'duration': duration,
            'action_type': action_type,
            'timestamp': time.time(),
            **velocity_result
        })
        self.velocity_cmd_pub.publish(velocity_msg)
        
        # Update statistics
        self.stats['total_commands'] += 1
        
        # Publish status update
        status_msg = String()
        status_msg.data = json.dumps({
            'velocity_command': self.current_velocity,
            'action_type': action_type,
            'timestamp': time.time(),
            'stats': self.stats
        })
        self.status_pub.publish(status_msg)
        
        logger.info(f"📤 Velocity command published: linear_x={linear_x:.3f}, angular_z={angular_z:.3f}")
    
    def control_loop(self):
        """Main control loop - called by ROS timer"""
        
        # Skip if already processing
        if not self.processing_lock.acquire(blocking=False):
            logger.debug("🔄 Skipping cycle - already processing")
            return
        
        try:
            self.is_processing = True
            
            # Get image directly from the camera
            logger.debug("📷 Acquiring image from RealSense camera...")
            cv_image = self._get_image_from_camera()

            if cv_image is None:
                logger.warning("⚠️ Failed to acquire image, skipping control cycle.")
                # The lock will be released in the 'finally' block.
                # No need to release it here, which caused the crash.
                return
            
            # Save the captured image to a file
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            milliseconds = int((time.time() - int(time.time())) * 1000)
            image_filename = f"img_{timestamp}_{milliseconds:03d}.jpg"
            image_path = os.path.join(self.image_log_dir, image_filename)
            try:
                cv2.imwrite(image_path, cv_image)
                logger.debug(f"💾 Image saved to {image_path}")
            except Exception as e:
                logger.error(f"❌ Failed to save image to {image_path}: {e}")
            
            # Send image to VLM server and get velocity command
            logger.debug("📤 Sending image to VLM server...")
            velocity_result = self._send_image_to_server(cv_image)
            
            if velocity_result:
                self._process_velocity_command(velocity_result)
            else:
                logger.warning("⚠️ No valid velocity command received")
                # Print stop command as fallback
                print("=" * 60)
                print("🛑 FALLBACK VELOCITY COMMAND (STOP):")
                print("   Linear X:  0.0000 m/s")
                print("   Angular Z: 0.0000 rad/s")
                print("   Duration:  0.8 seconds")
                print("   Action:    fallback_stop")
                print("=" * 60)
                
        except Exception as e:
            logger.error(f"💥 Control loop error: {e}")
            
        finally:
            self.is_processing = False
            # This ensures the lock is always released, but only once.
            if self.processing_lock.locked():
                self.processing_lock.release()
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive status summary"""
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
        """Graceful shutdown"""
        logger.info("🛑 Shutting down velocity controller...")

        # Stop the camera pipeline
        if hasattr(self, 'rs_pipeline'):
            logger.info("📷 Stopping RealSense pipeline...")
            self.rs_pipeline.stop()
        
        # Send final stop command
        print("=" * 60)
        print("🛑 SHUTDOWN VELOCITY COMMAND:")
        print("   Linear X:  0.0000 m/s")
        print("   Angular Z: 0.0000 rad/s")
        print("   Duration:  0.0 seconds")
        print("   Action:    shutdown_stop")
        print("=" * 60)
        
        # Cancel timer
        if hasattr(self, 'control_timer'):
            self.control_timer.cancel()
        
        logger.info("✅ Shutdown complete")


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    
    # Configuration - update these for your setup
    SERVER_IP = "10.165.232.223"  # VLM server internal IP
    SERVER_PORT = 8888
    INSTRUCTION = "Navigate to the door and stop in front of it"
    
    controller = None
    
    try:
        logger.info("🚀 Starting Go2 Velocity Controller...")
        
        controller = Go2VelocityController(
            server_ip=SERVER_IP,
            server_port=SERVER_PORT, 
            instruction=INSTRUCTION
        )
        
        logger.info("✅ Controller ready - starting velocity command processing")
        rclpy.spin(controller)
        
    except KeyboardInterrupt:
        logger.info("⌨️ Keyboard interrupt received")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
    finally:
        if controller:
            controller.shutdown()
        
        logger.info("🏁 Shutting down ROS2...")
        rclpy.shutdown() # shutdown ROS2


if __name__ == '__main__':
    main()