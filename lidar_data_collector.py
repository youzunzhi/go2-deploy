#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import threading
import time
import json
from sensor_msgs.msg import PointCloud2, Imu
from nav_msgs.msg import Odometry
import pickle

class LidarDataCollector(Node):
    def __init__(self):
        super().__init__('lidar_data_collector')
        
        # Data storage
        self.cloud_data = []
        self.cloud_base_data = []
        self.cloud_deskewed_data = []
        self.imu_data = []
        self.robot_odom_data = []
        
        # Subscribers
        self.cloud_sub = self.create_subscription(
            PointCloud2, '/utlidar/cloud', self.cloud_callback, 10)
        self.cloud_base_sub = self.create_subscription(
            PointCloud2, '/utlidar/cloud_base', self.cloud_base_callback, 10)
        self.cloud_deskewed_sub = self.create_subscription(
            PointCloud2, '/utlidar/cloud_deskewed', self.cloud_deskewed_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/utlidar/imu', self.imu_callback, 10)
        self.robot_odom_sub = self.create_subscription(
            Odometry, '/utlidar/robot_odom', self.robot_odom_callback, 10)
        
        self.get_logger().info('LiDAR Data Collector started. Collecting for 10 seconds...')
        
        # Start timer to stop after 10 seconds
        self.timer = self.create_timer(10.0, self.stop_collection)
        self.start_time = time.time()
        
    def cloud_callback(self, msg):
        timestamp = time.time() - self.start_time
        self.cloud_data.append({
            'timestamp': timestamp,
            'header': {
                'stamp': {'sec': msg.header.stamp.sec, 'nanosec': msg.header.stamp.nanosec},
                'frame_id': msg.header.frame_id
            },
            'height': msg.height,
            'width': msg.width,
            'fields': [{'name': f.name, 'offset': f.offset, 'datatype': f.datatype, 'count': f.count} for f in msg.fields],
            'is_bigendian': msg.is_bigendian,
            'point_step': msg.point_step,
            'row_step': msg.row_step,
            'data': list(msg.data),
            'is_dense': msg.is_dense
        })
        
    def cloud_base_callback(self, msg):
        timestamp = time.time() - self.start_time
        self.cloud_base_data.append({
            'timestamp': timestamp,
            'header': {
                'stamp': {'sec': msg.header.stamp.sec, 'nanosec': msg.header.stamp.nanosec},
                'frame_id': msg.header.frame_id
            },
            'height': msg.height,
            'width': msg.width,
            'fields': [{'name': f.name, 'offset': f.offset, 'datatype': f.datatype, 'count': f.count} for f in msg.fields],
            'is_bigendian': msg.is_bigendian,
            'point_step': msg.point_step,
            'row_step': msg.row_step,
            'data': list(msg.data),
            'is_dense': msg.is_dense
        })
        
    def cloud_deskewed_callback(self, msg):
        timestamp = time.time() - self.start_time
        self.cloud_deskewed_data.append({
            'timestamp': timestamp,
            'header': {
                'stamp': {'sec': msg.header.stamp.sec, 'nanosec': msg.header.stamp.nanosec},
                'frame_id': msg.header.frame_id
            },
            'height': msg.height,
            'width': msg.width,
            'fields': [{'name': f.name, 'offset': f.offset, 'datatype': f.datatype, 'count': f.count} for f in msg.fields],
            'is_bigendian': msg.is_bigendian,
            'point_step': msg.point_step,
            'row_step': msg.row_step,
            'data': list(msg.data),
            'is_dense': msg.is_dense
        })
        
    def imu_callback(self, msg):
        timestamp = time.time() - self.start_time
        self.imu_data.append({
            'timestamp': timestamp,
            'header': {
                'stamp': {'sec': msg.header.stamp.sec, 'nanosec': msg.header.stamp.nanosec},
                'frame_id': msg.header.frame_id
            },
            'orientation': {
                'x': msg.orientation.x,
                'y': msg.orientation.y,
                'z': msg.orientation.z,
                'w': msg.orientation.w
            },
            'orientation_covariance': list(msg.orientation_covariance),
            'angular_velocity': {
                'x': msg.angular_velocity.x,
                'y': msg.angular_velocity.y,
                'z': msg.angular_velocity.z
            },
            'angular_velocity_covariance': list(msg.angular_velocity_covariance),
            'linear_acceleration': {
                'x': msg.linear_acceleration.x,
                'y': msg.linear_acceleration.y,
                'z': msg.linear_acceleration.z
            },
            'linear_acceleration_covariance': list(msg.linear_acceleration_covariance)
        })
        
    def robot_odom_callback(self, msg):
        timestamp = time.time() - self.start_time
        self.robot_odom_data.append({
            'timestamp': timestamp,
            'header': {
                'stamp': {'sec': msg.header.stamp.sec, 'nanosec': msg.header.stamp.nanosec},
                'frame_id': msg.header.frame_id
            },
            'child_frame_id': msg.child_frame_id,
            'pose': {
                'pose': {
                    'position': {
                        'x': msg.pose.pose.position.x,
                        'y': msg.pose.pose.position.y,
                        'z': msg.pose.pose.position.z
                    },
                    'orientation': {
                        'x': msg.pose.pose.orientation.x,
                        'y': msg.pose.pose.orientation.y,
                        'z': msg.pose.pose.orientation.z,
                        'w': msg.pose.pose.orientation.w
                    }
                },
                'covariance': list(msg.pose.covariance)
            },
            'twist': {
                'twist': {
                    'linear': {
                        'x': msg.twist.twist.linear.x,
                        'y': msg.twist.twist.linear.y,
                        'z': msg.twist.twist.linear.z
                    },
                    'angular': {
                        'x': msg.twist.twist.angular.x,
                        'y': msg.twist.twist.angular.y,
                        'z': msg.twist.twist.angular.z
                    }
                },
                'covariance': list(msg.twist.covariance)
            }
        })
        
    def stop_collection(self):
        self.get_logger().info('10 seconds elapsed. Saving data to files...')
        
        # Save data to files
        with open('utlidar_cloud.json', 'w') as f:
            json.dump(self.cloud_data, f, indent=2)
            
        with open('utlidar_cloud_base.json', 'w') as f:
            json.dump(self.cloud_base_data, f, indent=2)
            
        with open('utlidar_cloud_deskewed.json', 'w') as f:
            json.dump(self.cloud_deskewed_data, f, indent=2)
            
        with open('utlidar_imu.json', 'w') as f:
            json.dump(self.imu_data, f, indent=2)
            
        with open('utlidar_robot_odom.json', 'w') as f:
            json.dump(self.robot_odom_data, f, indent=2)
            
        self.get_logger().info(f'Data saved:')
        self.get_logger().info(f'  Cloud messages: {len(self.cloud_data)}')
        self.get_logger().info(f'  Cloud base messages: {len(self.cloud_base_data)}')
        self.get_logger().info(f'  Cloud deskewed messages: {len(self.cloud_deskewed_data)}')
        self.get_logger().info(f'  IMU messages: {len(self.imu_data)}')
        self.get_logger().info(f'  Robot odom messages: {len(self.robot_odom_data)}')
        
        # Shutdown
        rclpy.shutdown()

def main():
    rclpy.init()
    collector = LidarDataCollector()
    
    try:
        rclpy.spin(collector)
    except KeyboardInterrupt:
        pass
    finally:
        collector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()