import rclpy
import torch
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import pyrealsense2 as rs
import torch.nn.functional as F

from utils.hardware_cfgs import ROS_TOPICS

@torch.no_grad()
def resize2d(img, size):
    return F.adaptive_avg_pool2d(img, size)

class RSDepthHandler:
    def __init__(self, output_resolution: tuple):
        """
        Args:
            output_resolution: [width, height] of the output depth image
        """
        self.rs_stream_width = 640
        self.rs_stream_height = 480
        self.rs_stream_fps = 30
        self.cropping = [60, 100, 80, 36] # top, bottom, left, right
        depth_range_m = [0.0, 3.0]
        self.depth_range = (depth_range_m[0] * 1000, depth_range_m[1] * 1000) # [m] -> [mm]
        self.output_resolution = output_resolution
        self.log_config()

        self.node = rclpy.create_node("rs_depth_handler")

        self.start_rs_pipeline()
        self.init_rs_filters()
        self.init_ros_communication()

    def start_rs_pipeline(self):
        self.rs_pipeline = rs.pipeline()
        self.rs_config = rs.config()
        self.rs_config.enable_stream(
            stream_type=rs.stream.depth,
            width=self.rs_stream_width,
            height=self.rs_stream_height,
            format=rs.format.z16, # 16-bit depth values in millimeters
            framerate=self.rs_stream_fps,
        )
        self.rs_pipeline.start(self.rs_config)
    
    def init_rs_filters(self):
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

    def init_ros_communication(self):
        self.forward_depth_image_pub = self.node.create_publisher(
            Float32MultiArray,
            ROS_TOPICS["DEPTH_IMAGE"],
            1,
        )
        self.log_info("rs_depth_handler initialized")

    def publish_depth_image(self):
        latency = 142 # ms
        rs_frame = self.rs_pipeline.wait_for_frames(latency)
        depth_frame = rs_frame.get_depth_frame()
        if not depth_frame:
            self.log_error("No depth frame", throttle_duration_sec=1)
            return
        
        for rs_filter in self.rs_filters:
            depth_frame = rs_filter.process(depth_frame)
        
        depth_image_tensor = torch.from_numpy(np.asanyarray(depth_frame.get_data()).astype(np.float32)).unsqueeze(0)

        depth_image_tensor = depth_image_tensor[:,
            self.cropping[0]: -self.cropping[1]-1,
            self.cropping[2]: -self.cropping[3]-1,
        ]

        depth_image_tensor = torch.clip(depth_image_tensor, self.depth_range[0], self.depth_range[1]) / (self.depth_range[1] - self.depth_range[0])
        depth_image_tensor = resize2d(depth_image_tensor, self.output_resolution)
        depth_image_tensor -= 0.5 # [-0.5, 0.5])

        depth_msg = Float32MultiArray()
        depth_msg.data = depth_image_tensor.flatten().detach().cpu().numpy().tolist()

        self.forward_depth_image_pub.publish(depth_msg)

    def log_config(self):
        self.log_info("rs_depth_handler initialized")
        self.log_info("rs_stream_width: {}".format(self.rs_stream_width))
        self.log_info("rs_stream_height: {}".format(self.rs_stream_height))
        self.log_info("rs_stream_fps: {}".format(self.rs_stream_fps))
        self.log_info("cropping: {}".format(self.cropping))
        self.log_info("depth_range: {}".format(self.depth_range))
        self.log_info("output_resolution: {}".format(self.output_resolution))

    def log_info(self, message, **kwargs):
        """Convenient logging method for info messages"""
        self.node.get_logger().info(message, **kwargs)
    
    def log_error(self, message, **kwargs):
        """Convenient logging method for error messages"""
        self.node.get_logger().error(message, **kwargs)