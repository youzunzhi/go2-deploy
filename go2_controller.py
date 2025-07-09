#!/usr/bin/env python3
"""
Go2 Velocity Command Controller
Receives velocity commands from NaVILA server and prints them for locomotion policy
"""

import io
import json
import time
import logging
from typing import Dict, Optional, Any
from threading import Lock

import cv2
import numpy as np
import requests
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge

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
        
        # ROS2 setup
        self.bridge = CvBridge()
        self._setup_ros_interfaces()
        
        # Robot state
        self.latest_image: Optional[Image] = None
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
    
    def _setup_ros_interfaces(self):
        """Setup ROS2 publishers and subscribers"""
        
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
        
        # Subscribers - support both raw and compressed images
        self.image_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            qos_profile
        )
        
        self.compressed_image_sub = self.create_subscription(
            CompressedImage,
            '/camera/color/image_raw/compressed',
            self.compressed_image_callback,
            qos_profile
        )
        
        logger.info("📡 ROS2 interfaces initialized")
    
    def image_callback(self, msg: Image):
        """Callback for raw image messages"""
        self.latest_image = msg
        self.stats['last_image_time'] = time.time()
    
    def compressed_image_callback(self, msg: CompressedImage):
        """Callback for compressed image messages"""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_image is not None:
                image_msg = self.bridge.cv2_to_imgmsg(cv_image, 'bgr8')
                image_msg.header = msg.header
                self.latest_image = image_msg
                self.stats['last_image_time'] = time.time()
        except Exception as e:
            logger.error(f"❌ Failed to process compressed image: {e}")
    
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
    
    def _send_image_to_server(self, image_msg: Image) -> Optional[Dict[str, Any]]:
        """Send image to VLM server and get velocity command"""
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            
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
        
        # Check if we have recent image data
        if self.latest_image is None:
            logger.warning("📷 No image data received yet")
            return
        
        # Check image age for safety
        image_age = time.time() - self.stats.get('last_image_time', 0)
        if image_age > 10.0:  # 10 second timeout
            logger.error(f"📷 Image data too old ({image_age:.1f}s)")
            return
        
        # Skip if already processing
        if not self.processing_lock.acquire(blocking=False):
            logger.debug("🔄 Skipping cycle - already processing")
            return
        
        try:
            self.is_processing = True
            
            # Send image to VLM server and get velocity command
            logger.debug("📤 Sending image to VLM server...")
            velocity_result = self._send_image_to_server(self.latest_image)
            
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
            self.processing_lock.release()
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive status summary"""
        current_time = time.time()
        
        return {
            'node_status': 'active',
            'server_url': self.server_url,
            'instruction': self.instruction,
            'processing': self.is_processing,
            'has_image': self.latest_image is not None,
            'image_age': current_time - self.stats.get('last_image_time', 0),
            'current_velocity': self.current_velocity,
            'last_velocity_time': self.last_velocity_time,
            'stats': self.stats.copy()
        }
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Shutting down velocity controller...")
        
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
    SERVER_IP = "192.168.1.100"  # VLM server IP
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
        rclpy.shutdown()


if __name__ == '__main__':
    main()