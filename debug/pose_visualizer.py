#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from scipy.spatial.transform import Rotation as R


class PoseVisualizer(Node):
    def __init__(self):
        super().__init__('pose_visualizer')
        
        self.subscription = self.create_subscription(
            Odometry,
            '/utlidar/robot_odom',
            self.pose_callback,
            10
        )
        
        self.poses = []
        self.start_time = time.time()
        self.get_logger().info('Pose visualizer started. Collecting data...')
        
    def pose_callback(self, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        
        pose_data = {
            'timestamp': time.time() - self.start_time,
            'position': [position.x, position.y, position.z],
            'orientation': [orientation.x, orientation.y, orientation.z, orientation.w]
        }
        
        self.poses.append(pose_data)
        
        if len(self.poses) % 50 == 0:
            self.get_logger().info(f'Collected {len(self.poses)} poses')
    
    def save_and_visualize(self, duration=30, filename='pose_trajectory'):
        self.get_logger().info(f'Starting data collection for {duration} seconds...')
        
        start_time = time.time()
        while (time.time() - start_time) < duration:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        self.get_logger().info(f'Data collection complete. Total poses: {len(self.poses)}')
        
        with open(f'{filename}.pkl', 'wb') as f:
            pickle.dump(self.poses, f)
        self.get_logger().info(f'Data saved to {filename}.pkl')
        
        self.create_3d_visualization(filename)
    
    def create_3d_visualization(self, filename):
        if not self.poses:
            self.get_logger().warning('No pose data to visualize')
            return
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        positions = np.array([pose['position'] for pose in self.poses])
        orientations = np.array([pose['orientation'] for pose in self.poses])
        
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        
        ax.plot(x, y, z, 'b-', alpha=0.6, linewidth=1, label='Trajectory')
        
        step = max(1, len(self.poses) // 20)
        for i in range(0, len(self.poses), step):
            pos = positions[i]
            quat = orientations[i]
            
            r = R.from_quat([quat[0], quat[1], quat[2], quat[3]])
            rotation_matrix = r.as_matrix()
            
            arrow_length = 0.05
            
            x_axis = rotation_matrix[:, 0] * arrow_length
            y_axis = rotation_matrix[:, 1] * arrow_length
            z_axis = rotation_matrix[:, 2] * arrow_length
            
            ax.quiver(pos[0], pos[1], pos[2], 
                     x_axis[0], x_axis[1], x_axis[2], 
                     color='red', alpha=0.8, arrow_length_ratio=0.3)
            
            ax.quiver(pos[0], pos[1], pos[2], 
                     y_axis[0], y_axis[1], y_axis[2], 
                     color='green', alpha=0.8, arrow_length_ratio=0.3)
            
            ax.quiver(pos[0], pos[1], pos[2], 
                     z_axis[0], z_axis[1], z_axis[2], 
                     color='blue', alpha=0.8, arrow_length_ratio=0.3)
        
        ax.scatter(x[0], y[0], z[0], color='green', s=100, label='Start', marker='o')
        ax.scatter(x[-1], y[-1], z[-1], color='red', s=100, label='End', marker='s')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Robot Pose Trajectory with Orientation Arrows\n(Red=X, Green=Y, Blue=Z axes)')
        ax.legend()
        
        ax.set_box_aspect([1,1,0.5])
        
        x_range = max(x) - min(x)
        y_range = max(y) - min(y)
        z_range = max(z) - min(z)
        
        center_x, center_y, center_z = np.mean(x), np.mean(y), np.mean(z)
        
        min_range = 0.25
        x_half_range = max(x_range / 2, min_range)
        y_half_range = max(y_range / 2, min_range)
        z_half_range = max(z_range / 2, min_range)
        
        ax.set_xlim(center_x - x_half_range, center_x + x_half_range)
        ax.set_ylim(center_y - y_half_range, center_y + y_half_range)
        ax.set_zlim(center_z - z_half_range, center_z + z_half_range)
        
        plt.tight_layout()
        plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
        self.get_logger().info(f'Visualization saved to {filename}.png')
        
        stats_text = f"""
Trajectory Statistics:
- Total poses: {len(self.poses)}
- Duration: {self.poses[-1]['timestamp']:.2f} seconds
- X range: {x_range:.4f} m
- Y range: {y_range:.4f} m  
- Z range: {z_range:.4f} m
- Average height: {np.mean(z):.4f} m
        """
        print(stats_text)


def main(args=None):
    rclpy.init(args=args)
    
    visualizer = PoseVisualizer()
    
    try:
        visualizer.save_and_visualize(duration=30, filename='robot_pose_trajectory')
    except KeyboardInterrupt:
        visualizer.get_logger().info('Data collection interrupted by user')
        if visualizer.poses:
            visualizer.create_3d_visualization('robot_pose_trajectory_partial')
    finally:
        visualizer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()