import threading
import torch
import numpy as np
import pyrealsense2 as rs
import torch.nn.functional as F

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from utils.hardware_cfgs import ROS_TOPICS


@torch.no_grad()
def resize2d(img, size):
    return F.adaptive_avg_pool2d(img, size)


class DepthCaptureHandler:
    """
    Handles RealSense depth camera capture and processing logic.
    Runs in separate thread to avoid blocking ROS operations.
    """
    
    def __init__(self, output_resolution: tuple):
        """
        Args:
            output_resolution: (width, height) for output depth image 
        """
        self.output_resolution = output_resolution
        
        # RealSense configuration (from original RSDepthHandler)
        self.rs_stream_width = 640
        self.rs_stream_height = 480
        self.rs_stream_fps = 30
        self.cropping = [60, 100, 80, 36]  # top, bottom, left, right
        depth_range_m = [0.0, 3.0]
        self.depth_range = (depth_range_m[0] * 1000, depth_range_m[1] * 1000)  # [m] -> [mm]
        
        # State
        self.is_running = False
        self.capture_thread = None
        self.latest_depth_tensor = None
        self.tensor_lock = threading.Lock()
        
        # Initialize RealSense pipeline and filters
        self._start_rs_pipeline()
        self._init_rs_filters()

    def _start_rs_pipeline(self):
        """Initialize RealSense pipeline"""
        self.rs_pipeline = rs.pipeline()
        self.rs_config = rs.config()
        self.rs_config.enable_stream(
            stream_type=rs.stream.depth,
            width=self.rs_stream_width,
            height=self.rs_stream_height,
            format=rs.format.z16,  # 16-bit depth values in millimeters
            framerate=self.rs_stream_fps,
        )
        self.rs_pipeline.start(self.rs_config)
    
    def _init_rs_filters(self):
        """Initialize RealSense filters for depth processing"""
        # Fill missing depth pixels (holes) in the depth image
        rs_hole_filling_filter = rs.hole_filling_filter()

        # Spatial filter reduces noise by smoothing neighboring pixels
        rs_spatial_filter = rs.spatial_filter()
        rs_spatial_filter.set_option(rs.option.filter_magnitude, 5)  # Smoothing strength (0-5)
        rs_spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.75)  # Edge preservation (0-1)
        rs_spatial_filter.set_option(rs.option.filter_smooth_delta, 1)  # Depth difference threshold
        rs_spatial_filter.set_option(rs.option.holes_fill, 4)  # Hole filling aggressiveness (0-5)

        # Temporal filter reduces noise by comparing frames over time
        rs_temporal_filter = rs.temporal_filter()
        rs_temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.75)  # Temporal smoothing strength
        rs_temporal_filter.set_option(rs.option.filter_smooth_delta, 1)  # Frame difference threshold

        # Apply filters in order: hole filling -> spatial -> temporal
        self.rs_filters = [
            rs_hole_filling_filter,
            rs_spatial_filter,
            rs_temporal_filter,
        ]
    
    def _capture_loop(self, logger=None):
        """Main capture loop running in separate thread"""
        latency = 142  # ms - timeout for waiting for frames
        
        while self.is_running:
            try:
                # Capture depth frame (this is the blocking call)
                rs_frame = self.rs_pipeline.wait_for_frames(latency)
                depth_frame = rs_frame.get_depth_frame()
                
                if not depth_frame:
                    if logger:
                        logger.warn("No depth frame received")
                    continue
                
                # Apply filters
                for rs_filter in self.rs_filters:
                    depth_frame = rs_filter.process(depth_frame)
                
                # Convert to tensor and process
                depth_image_tensor = self._process_depth_frame(depth_frame)
                
                # Store latest tensor with thread safety
                with self.tensor_lock:
                    self.latest_depth_tensor = depth_image_tensor
                    
            except Exception as e:
                if logger:
                    logger.error(f"Error in depth capture loop: {e}")
                    logger.error("Sending error signal and stopping depth capture")
                
                # Send error signal by setting tensor to None
                with self.tensor_lock:
                    self.latest_depth_tensor = None
                
                self.is_running = False
                break
                
    def start_capture(self, logger=None):
        """Start depth image capture in separate thread"""
        if self.is_running:
            if logger:
                logger.warn("Depth capture is already running")
            return
            
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, args=(logger,), daemon=True)
        self.capture_thread.start()
        if logger:
            logger.info("Depth capture thread started")

    def stop_capture(self, logger=None):
        """Stop depth image capture"""
        if not self.is_running:
            return
            
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        if logger:
            logger.info("Depth capture thread stopped")

    def get_latest_tensor(self):
        """Get latest depth tensor (thread-safe)"""
        with self.tensor_lock:
            if self.latest_depth_tensor is None:
                return None
            return self.latest_depth_tensor.clone()

    def _process_depth_frame(self, depth_frame):
        """Process depth frame into tensor (same logic as original RSDepthHandler)"""
        # Convert to numpy array and then tensor
        depth_image_tensor = torch.from_numpy(
            np.asanyarray(depth_frame.get_data()).astype(np.float32)
        ).unsqueeze(0)

        # Apply cropping
        depth_image_tensor = depth_image_tensor[:,
            self.cropping[0]: -self.cropping[1]-1,
            self.cropping[2]: -self.cropping[3]-1,
        ]

        # Clip to depth range and normalize
        depth_image_tensor = torch.clip(
            depth_image_tensor, 
            self.depth_range[0], 
            self.depth_range[1]
        ) / (self.depth_range[1] - self.depth_range[0])
        
        # Resize to output resolution
        depth_image_tensor = resize2d(depth_image_tensor, self.output_resolution)
        
        # Center around 0 ([-0.5, 0.5])
        depth_image_tensor -= 0.5
        
        return depth_image_tensor

    def shutdown(self):
        """Clean shutdown"""
        self.stop_capture()
        if hasattr(self, 'rs_pipeline'):
            self.rs_pipeline.stop()

    def log_config(self, logger):
        """Log configuration parameters"""
        logger.info("DepthCaptureHandler configuration:")
        logger.info(f"  rs_stream_width: {self.rs_stream_width}")
        logger.info(f"  rs_stream_height: {self.rs_stream_height}")
        logger.info(f"  rs_stream_fps: {self.rs_stream_fps}")
        logger.info(f"  cropping: {self.cropping}")
        logger.info(f"  depth_range: {self.depth_range}")
        logger.info(f"  output_resolution: {self.output_resolution}")


class DepthImagePublisherNode(Node):
    """
    ROS2 node that publishes depth image tensors received from DepthCaptureHandler.
    Focuses solely on ROS communication and publishing logic.
    """
    
    def __init__(self, capture_handler: DepthCaptureHandler):
        """
        Args:
            capture_handler: DepthCaptureHandler instance for getting depth data
            publish_rate_hz: Rate to publish depth tensors (Hz)
        """
        super().__init__('depth_image_publisher')
        
        self.capture_handler = capture_handler
        self.publish_rate_hz = 100.0
        self.topic_name = ROS_TOPICS["DEPTH_IMAGE"]
        
        # Create publisher for depth image tensors
        self.depth_publisher = self.create_publisher(
            Float32MultiArray,
            self.topic_name,
            10  # queue size
        )
        
        # Create timer for publishing at specified rate
        self.publish_timer = self.create_timer(
            1.0 / self.publish_rate_hz,
            self.publish_latest_depth
        )
        
        self.get_logger().info(f"DepthImagePublisherNode initialized, publishing to {self.topic_name} at {self.publish_rate_hz} Hz")

    def publish_latest_depth(self):
        """Publish the latest depth tensor to ROS2 topic"""
        tensor_to_publish = self.capture_handler.get_latest_tensor()
        
        # Create Float32MultiArray message
        msg = Float32MultiArray()
        
        if tensor_to_publish is None:
            # Send error signal - empty data array will cause failure in subscriber
            msg.data = []
            self.get_logger().warn("Publishing empty depth tensor due to capture error")
        else:
            # Normal case - flatten tensor to 1D list
            msg.data = tensor_to_publish.flatten().tolist()
        
        # Publish (either normal data or error signal)
        self.depth_publisher.publish(msg)


class DepthImagePublisherRunner:
    """
    Runner class that manages both DepthCaptureHandler and DepthImagePublisherNode.
    Provides clean separation of concerns and lifecycle management.
    """
    
    def __init__(self, output_resolution: tuple):
        """
        Args:
            output_resolution: (width, height) for output depth image
        """
        self.output_resolution = output_resolution
        
        # Create capture handler
        self.capture_handler = DepthCaptureHandler(output_resolution)
        
        # Create ROS node
        self.node = DepthImagePublisherNode(self.capture_handler)
        
        # Log configuration
        self.capture_handler.log_config(self.node.get_logger())

    def start(self):
        """Start depth capture and ROS node"""
        # Start capture thread
        self.capture_handler.start_capture(self.node.get_logger())
        
        # Node timer starts automatically when created
        self.node.get_logger().info("DepthImagePublisherRunner started successfully")

    def spin(self):
        """Run ROS node (blocking)"""
        try:
            rclpy.spin(self.node)
        finally:
            self.shutdown()

    def shutdown(self):
        """Clean shutdown of all components"""
        self.capture_handler.shutdown()
        if hasattr(self.node, 'destroy_node'):
            self.node.destroy_node()


def run_depth_publisher_process(output_resolution: tuple):
    """
    Function to run depth publisher node in a separate process
    
    Args:
        output_resolution: (width, height) for output depth image
        publish_rate_hz: Rate to publish depth tensors (Hz)
    """
    rclpy.init()
    
    try:
        # Create and start runner
        runner = DepthImagePublisherRunner(output_resolution)
        runner.start()
        
        # Run ROS node (blocking)
        runner.spin()
        
    finally:
        rclpy.shutdown()