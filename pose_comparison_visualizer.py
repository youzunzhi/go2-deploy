#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.spatial.transform import Rotation as R

class PoseComparisonVisualizer(Node):
    def __init__(self):
        super().__init__('pose_comparison_visualizer')
        
        self.utlidar_subscription = self.create_subscription(
            Odometry,
            '/utlidar/robot_odom',
            self.utlidar_pose_callback,
            10
        )
        
        self.filtered_subscription = self.create_subscription(
            Odometry,
            '/odometry/filtered',
            self.filtered_pose_callback,
            10
        )
        
        self.utlidar_poses = []
        self.filtered_poses = []
        self.start_time = time.time()
        self.get_logger().info('Pose comparison visualizer started. Collecting data...')
        
    def utlidar_pose_callback(self, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        
        pose_data = {
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            'position': np.array([position.x, position.y, position.z]),
            'orientation': np.array([orientation.x, orientation.y, orientation.z, orientation.w])
        }
        
        self.utlidar_poses.append(pose_data)
        
        if len(self.utlidar_poses) % 50 == 0:
            self.get_logger().info(f'Collected {len(self.utlidar_poses)} utlidar poses')

    def filtered_pose_callback(self, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        
        pose_data = {
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            'position': np.array([position.x, position.y, position.z]),
            'orientation': np.array([orientation.x, orientation.y, orientation.z, orientation.w])
        }
        
        self.filtered_poses.append(pose_data)

        if len(self.filtered_poses) % 50 == 0:
            self.get_logger().info(f'Collected {len(self.filtered_poses)} filtered poses')

    def save_and_visualize(self, duration=30):
        self.get_logger().info(f'Starting data collection for {duration} seconds...')
        
        start_time = time.time()
        while (time.time() - start_time) < duration:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        self.get_logger().info(f'Data collection complete.')
        self.get_logger().info(f'Total utlidar poses: {len(self.utlidar_poses)}')
        self.get_logger().info(f'Total filtered poses: {len(self.filtered_poses)}')

        if self.utlidar_poses:
            self.create_3d_visualization(self.utlidar_poses, 'utlidar_robot_odom_trajectory')
        if self.filtered_poses:
            self.create_3d_visualization(self.filtered_poses, 'odometry_filtered_trajectory')
        if self.utlidar_poses and self.filtered_poses:
            self.create_comparison_visualization('comparison_trajectory')
            self.calculate_and_print_diff()

    def create_3d_visualization(self, poses, filename):
        if not poses:
            self.get_logger().warning(f'No pose data to visualize for {filename}')
            return
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        positions = np.array([pose['position'] for pose in poses])
        orientations = np.array([pose['orientation'] for pose in poses])
        
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        
        ax.plot(x, y, z, 'b-', alpha=0.6, linewidth=1, label='Trajectory')
        
        step = max(1, len(poses) // 20)
        for i in range(0, len(poses), step):
            pos = positions[i]
            quat = orientations[i]
            
            r = R.from_quat(quat)
            rotation_matrix = r.as_matrix()
            
            arrow_length = 0.05
            
            x_axis = rotation_matrix[:, 0] * arrow_length
            y_axis = rotation_matrix[:, 1] * arrow_length
            z_axis = rotation_matrix[:, 2] * arrow_length
            
            ax.quiver(pos[0], pos[1], pos[2], x_axis[0], x_axis[1], x_axis[2], color='red', alpha=0.8, arrow_length_ratio=0.3)
            ax.quiver(pos[0], pos[1], pos[2], y_axis[0], y_axis[1], y_axis[2], color='green', alpha=0.8, arrow_length_ratio=0.3)
            ax.quiver(pos[0], pos[1], pos[2], z_axis[0], z_axis[1], z_axis[2], color='blue', alpha=0.8, arrow_length_ratio=0.3)
        
        ax.scatter(x[0], y[0], z[0], color='green', s=100, label='Start', marker='o')
        ax.scatter(x[-1], y[-1], z[-1], color='red', s=100, label='End', marker='s')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Robot Pose Trajectory: {filename}\n(Red=X, Green=Y, Blue=Z axes)')
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

    def create_comparison_visualization(self, filename):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot utlidar trajectory
        utlidar_positions = np.array([pose['position'] for pose in self.utlidar_poses])
        x_u, y_u, z_u = utlidar_positions[:, 0], utlidar_positions[:, 1], utlidar_positions[:, 2]
        ax.plot(x_u, y_u, z_u, 'b-', alpha=0.7, linewidth=2, label='/utlidar/robot_odom')

        # Plot filtered trajectory
        filtered_positions = np.array([pose['position'] for pose in self.filtered_poses])
        x_f, y_f, z_f = filtered_positions[:, 0], filtered_positions[:, 1], filtered_positions[:, 2]
        ax.plot(x_f, y_f, z_f, 'r-', alpha=0.7, linewidth=2, label='/odometry/filtered')

        ax.scatter(x_u[0], y_u[0], z_u[0], color='blue', s=150, marker='o', label='Start /utlidar/robot_odom')
        ax.scatter(x_u[-1], y_u[-1], z_u[-1], color='blue', s=150, marker='s', label='End /utlidar/robot_odom')
        ax.scatter(x_f[0], y_f[0], z_f[0], color='red', s=150, marker='o', label='Start /odometry/filtered')
        ax.scatter(x_f[-1], y_f[-1], z_f[-1], color='red', s=150, marker='s', label='End /odometry/filtered')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Comparison of Robot Pose Trajectories')
        ax.legend()

        all_x = np.concatenate([x_u, x_f])
        all_y = np.concatenate([y_u, y_f])
        all_z = np.concatenate([z_u, z_f])

        x_range = max(all_x) - min(all_x)
        y_range = max(all_y) - min(all_y)
        z_range = max(all_z) - min(all_z)
        
        center_x, center_y, center_z = np.mean(all_x), np.mean(all_y), np.mean(all_z)
        
        min_range = 0.25
        x_half_range = max(x_range / 2, min_range)
        y_half_range = max(y_range / 2, min_range)
        z_half_range = max(z_range / 2, min_range)
        
        ax.set_xlim(center_x - x_half_range, center_x + x_half_range)
        ax.set_ylim(center_y - y_half_range, center_y + y_half_range)
        ax.set_zlim(center_z - z_half_range, center_z + z_half_range)

        ax.set_box_aspect([1,1,0.5])
        plt.tight_layout()
        plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
        self.get_logger().info(f'Comparison visualization saved to {filename}.png')

    def calculate_and_print_diff(self):
        if not self.utlidar_poses or not self.filtered_poses:
            self.get_logger().warning("Not enough data to calculate differences.")
            return

        utlidar_ts = np.array([p['timestamp'] for p in self.utlidar_poses])
        
        pos_diffs = []
        rpy_diffs = []

        for filtered_pose in self.filtered_poses:
            filtered_ts = filtered_pose['timestamp']
            
            # Find closest utlidar pose by timestamp
            time_diffs = np.abs(utlidar_ts - filtered_ts)
            closest_idx = np.argmin(time_diffs)
            
            # only compare if timestamps are close enough (e.g., within 50ms)
            if time_diffs[closest_idx] > 0.05:
                continue

            utlidar_pose = self.utlidar_poses[closest_idx]

            # Position difference
            pos_diff = filtered_pose['position'] - utlidar_pose['position']
            pos_diffs.append(pos_diff)

            # Orientation difference
            r_filtered = R.from_quat(filtered_pose['orientation'])
            r_utlidar = R.from_quat(utlidar_pose['orientation'])
            
            rpy_filtered = r_filtered.as_euler('xyz', degrees=True)
            rpy_utlidar = r_utlidar.as_euler('xyz', degrees=True)
            
            rpy_diff = rpy_filtered - rpy_utlidar
            # Handle angle wrapping
            rpy_diff = (rpy_diff + 180) % 360 - 180
            rpy_diffs.append(rpy_diff)

        if not pos_diffs:
            self.get_logger().warning("Could not find any matching poses by timestamp to calculate diff.")
            return

        pos_diffs = np.array(pos_diffs)
        rpy_diffs = np.array(rpy_diffs)

        abs_pos_diffs = np.abs(pos_diffs)
        abs_rpy_diffs = np.abs(rpy_diffs)

        mean_pos_diff = np.mean(abs_pos_diffs, axis=0)
        max_pos_diff = np.max(abs_pos_diffs, axis=0)
        
        mean_rpy_diff = np.mean(abs_rpy_diffs, axis=0)
        max_rpy_diff = np.max(abs_rpy_diffs, axis=0)

        stats_text = f"""
Pose Difference Statistics:
- Number of compared pose pairs: {len(pos_diffs)}

Position Difference (meters):
- Mean Diff (X, Y, Z): {mean_pos_diff[0]:.4f}, {mean_pos_diff[1]:.4f}, {mean_pos_diff[2]:.4f}
- Max Diff (X, Y, Z):  {max_pos_diff[0]:.4f}, {max_pos_diff[1]:.4f}, {max_pos_diff[2]:.4f}

Orientation Difference (degrees):
- Mean Diff (Roll, Pitch, Yaw): {mean_rpy_diff[0]:.4f}, {mean_rpy_diff[1]:.4f}, {mean_rpy_diff[2]:.4f}
- Max Diff (Roll, Pitch, Yaw):  {max_rpy_diff[0]:.4f}, {max_rpy_diff[1]:.4f}, {max_rpy_diff[2]:.4f}
        """
        print(stats_text)


def main(args=None):
    rclpy.init(args=args)
    
    visualizer = PoseComparisonVisualizer()
    
    try:
        visualizer.save_and_visualize(duration=10)
    except KeyboardInterrupt:
        visualizer.get_logger().info('Data collection interrupted by user')
        if visualizer.utlidar_poses or visualizer.filtered_poses:
            visualizer.get_logger().info('Creating visualizations from partial data...')
            if visualizer.utlidar_poses:
                visualizer.create_3d_visualization(visualizer.utlidar_poses, 'utlidar_robot_odom_trajectory_partial')
            if visualizer.filtered_poses:
                visualizer.create_3d_visualization(visualizer.filtered_poses, 'odometry_filtered_trajectory_partial')
            if visualizer.utlidar_poses and visualizer.filtered_poses:
                visualizer.create_comparison_visualization('comparison_trajectory_partial')
                visualizer.calculate_and_print_diff()
    finally:
        visualizer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
