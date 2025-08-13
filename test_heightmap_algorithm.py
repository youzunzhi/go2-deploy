#!/usr/bin/env python3
"""
Test Height Map Algorithm with Synthetic Data

This script tests the height map processing algorithm using synthetic point cloud data.
It's useful for validating the algorithm without requiring actual LiDAR hardware.

Usage:
    python test_heightmap_algorithm.py
    python test_heightmap_algorithm.py --scenario stairs
    python test_heightmap_algorithm.py --scenario terrain
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
from collections import deque
from scipy.ndimage import maximum_filter

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import existing components
from lidar_height_map_processor import LiDARHeightMapProcessor

class SyntheticPointCloudGenerator:
    """
    生成合成点云数据用于测试height map算法
    """
    
    def __init__(self):
        # LiDAR参数 (基于Unitree L1规格)
        self.fov_horizontal = 360  # 度
        self.fov_vertical = 90     # 度
        self.max_range = 30        # 米
        self.angular_resolution = 0.2  # 度
        
        # Height map范围 (与processor保持一致)
        self.x_range = [-0.8, 0.2]  # 1m前向范围
        self.y_range = [-0.8, 0.8]  # 1.6m横向范围
        self.z_range = [0.0, 5.0]   # 5m垂直范围
    
    def generate_flat_ground(self, noise_level=0.01):
        """生成平地点云"""
        points = []
        
        # 在height map范围内生成点
        x_points = np.linspace(self.x_range[0], self.x_range[1], 100)
        y_points = np.linspace(self.y_range[0], self.y_range[1], 200)
        
        for x in x_points:
            for y in y_points:
                if np.random.random() < 0.8:  # 80%的点被检测到
                    z = 0.0 + np.random.normal(0, noise_level)  # 添加噪声
                    points.append([x, y, z])
        
        return np.array(points)
    
    def generate_stairs(self, step_height=0.15, step_depth=0.3, num_steps=5):
        """生成楼梯点云"""
        points = []
        
        # 楼梯从x=0开始向前
        current_x = 0.0
        current_z = 0.0
        
        for step in range(num_steps):
            # 每个台阶的点云
            step_x_range = [current_x, current_x + step_depth]
            
            if step_x_range[0] < self.x_range[1]:  # 在范围内
                y_points = np.linspace(self.y_range[0], self.y_range[1], 50)
                x_points = np.linspace(max(step_x_range[0], self.x_range[0]), 
                                     min(step_x_range[1], self.x_range[1]), 20)
                
                for x in x_points:
                    for y in y_points:
                        if np.random.random() < 0.7:
                            z = current_z + np.random.normal(0, 0.01)
                            if self.z_range[0] <= z <= self.z_range[1]:
                                points.append([x, y, z])
            
            current_x += step_depth
            current_z += step_height
        
        # 添加起始平地
        ground_points = self.generate_flat_ground(noise_level=0.01)
        ground_mask = ground_points[:, 0] < 0.0  # x < 0的区域保持平地
        ground_points = ground_points[ground_mask]
        
        if len(points) > 0:
            all_points = np.vstack([ground_points, np.array(points)])
        else:
            all_points = ground_points
        
        return all_points
    
    def generate_rough_terrain(self, amplitude=0.3, frequency=3.0):
        """生成起伏地形点云"""
        points = []
        
        x_points = np.linspace(self.x_range[0], self.x_range[1], 150)
        y_points = np.linspace(self.y_range[0], self.y_range[1], 250)
        
        for x in x_points:
            for y in y_points:
                if np.random.random() < 0.75:  # 75%的点被检测到
                    # 使用正弦波生成起伏地形
                    z = (amplitude * np.sin(frequency * x) * np.cos(frequency * y * 0.7) + 
                         0.1 * np.sin(10 * x) * np.sin(8 * y) +  # 高频细节
                         np.random.normal(0, 0.02))  # 噪声
                    
                    # 确保在Z范围内
                    z = max(self.z_range[0], min(z, self.z_range[1]))
                    points.append([x, y, z])
        
        return np.array(points)
    
    def generate_obstacles(self, num_obstacles=5, max_height=0.5):
        """生成包含障碍物的地形"""
        # 先生成平地
        points = self.generate_flat_ground(noise_level=0.01).tolist()
        
        # 添加随机障碍物
        for _ in range(num_obstacles):
            # 随机障碍物位置和尺寸
            center_x = np.random.uniform(self.x_range[0] + 0.2, self.x_range[1] - 0.2)
            center_y = np.random.uniform(self.y_range[0] + 0.2, self.y_range[1] - 0.2)
            width = np.random.uniform(0.1, 0.3)
            length = np.random.uniform(0.1, 0.3)
            height = np.random.uniform(0.1, max_height)
            
            # 生成障碍物点云
            obstacle_x = np.linspace(center_x - width/2, center_x + width/2, 20)
            obstacle_y = np.linspace(center_y - length/2, center_y + length/2, 20)
            
            for x in obstacle_x:
                for y in obstacle_y:
                    if (self.x_range[0] <= x <= self.x_range[1] and 
                        self.y_range[0] <= y <= self.y_range[1]):
                        z = height + np.random.normal(0, 0.01)
                        points.append([x, y, z])
        
        return np.array(points)

class HeightMapTester:
    """
    测试height map算法的主类
    """
    
    def __init__(self):
        self.lidar_processor = LiDARHeightMapProcessor(device="cpu")
        self.point_generator = SyntheticPointCloudGenerator()
        
    def test_scenario(self, scenario_name):
        """测试特定场景"""
        print(f"Testing scenario: {scenario_name}")
        
        # 生成点云数据
        if scenario_name == "flat":
            points = self.point_generator.generate_flat_ground()
        elif scenario_name == "stairs":
            points = self.point_generator.generate_stairs()
        elif scenario_name == "terrain":
            points = self.point_generator.generate_rough_terrain()
        elif scenario_name == "obstacles":
            points = self.point_generator.generate_obstacles()
        else:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        print(f"Generated {len(points)} points")
        
        # 创建模拟的PointCloud2消息
        mock_msg = self._create_mock_pointcloud_msg(points)
        
        # 处理多帧数据测试maximum filter
        heightmaps = []
        for frame in range(5):
            # 每帧添加少量噪声模拟真实情况
            noisy_points = points + np.random.normal(0, 0.01, points.shape)
            mock_msg = self._create_mock_pointcloud_msg(noisy_points)
            
            heightmap = self.lidar_processor.process_pointcloud_to_heightmap(mock_msg)
            heightmaps.append(heightmap)
            print(f"Frame {frame+1}: Generated heightmap with shape {heightmap.shape}")
        
        # 返回最后一个heightmap (已应用maximum filter)和原始点云
        return heightmaps[-1], points
    
    def _create_mock_pointcloud_msg(self, points):
        """创建模拟的PointCloud2消息"""
        class MockPointCloud2:
            def __init__(self, points):
                # 将点云数据编码为字节
                self.data = []
                self.point_step = 12  # 每个点3个float32 = 12字节
                
                for point in points:
                    # 每个坐标打包为float32
                    import struct
                    self.data.extend(struct.pack('fff', point[0], point[1], point[2]))
        
        return MockPointCloud2(points)
    
    def visualize_results(self, heightmap_tensor, points, scenario_name):
        """可视化结果"""
        # 转换heightmap为2D数组
        heightmap_flat = heightmap_tensor.cpu().numpy().flatten()
        heightmap_2d = heightmap_flat.reshape(self.lidar_processor.y_bins, 
                                            self.lidar_processor.x_bins)
        
        # 创建可视化
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Height Map Algorithm Test - {scenario_name.capitalize()}', 
                    fontsize=16)
        
        # 原始点云 - 顶视图
        axes[0, 0].scatter(points[:, 0], points[:, 1], c=points[:, 2], 
                          cmap='viridis', s=1, alpha=0.6)
        axes[0, 0].set_title('Original Point Cloud (Top View)')
        axes[0, 0].set_xlabel('X (m)')
        axes[0, 0].set_ylabel('Y (m)')
        axes[0, 0].set_xlim(self.lidar_processor.range_x)
        axes[0, 0].set_ylim(self.lidar_processor.range_y)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 原始点云 - 侧视图
        axes[0, 1].scatter(points[:, 0], points[:, 2], c=points[:, 1], 
                          cmap='plasma', s=1, alpha=0.6)
        axes[0, 1].set_title('Original Point Cloud (Side View)')
        axes[0, 1].set_xlabel('X (m)')
        axes[0, 1].set_ylabel('Z (m)')
        axes[0, 1].set_xlim(self.lidar_processor.range_x)
        axes[0, 1].set_ylim(self.lidar_processor.range_z)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Height map结果
        im = axes[1, 0].imshow(heightmap_2d, cmap='terrain', origin='lower', 
                              interpolation='bilinear')
        axes[1, 0].set_title('Generated Height Map (17×27 Grid)')
        
        # 设置正确的坐标轴标签
        x_ticks = np.linspace(0, self.lidar_processor.x_bins-1, 5)
        y_ticks = np.linspace(0, self.lidar_processor.y_bins-1, 5)
        x_labels = np.linspace(self.lidar_processor.range_x[0], 
                              self.lidar_processor.range_x[1], 5)
        y_labels = np.linspace(self.lidar_processor.range_y[0], 
                              self.lidar_processor.range_y[1], 5)
        
        axes[1, 0].set_xticks(x_ticks)
        axes[1, 0].set_yticks(y_ticks)
        axes[1, 0].set_xticklabels([f'{x:.1f}' for x in x_labels])
        axes[1, 0].set_yticklabels([f'{y:.1f}' for y in y_labels])
        axes[1, 0].set_xlabel('X (m)')
        axes[1, 0].set_ylabel('Y (m)')
        
        plt.colorbar(im, ax=axes[1, 0], label='Height (m)')
        
        # 添加网格显示voxel边界
        for i in range(self.lidar_processor.x_bins + 1):
            axes[1, 0].axvline(i - 0.5, color='white', alpha=0.2, linewidth=0.5)
        for i in range(self.lidar_processor.y_bins + 1):
            axes[1, 0].axhline(i - 0.5, color='white', alpha=0.2, linewidth=0.5)
        
        # 统计信息
        valid_heights = heightmap_flat[heightmap_flat > -9.0]
        axes[1, 1].hist(valid_heights, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(np.mean(valid_heights), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(valid_heights):.3f}m')
        axes[1, 1].axvline(np.median(valid_heights), color='orange', linestyle='--', 
                          label=f'Median: {np.median(valid_heights):.3f}m')
        axes[1, 1].set_title('Height Distribution')
        axes[1, 1].set_xlabel('Height (m)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 添加算法信息文本
        algorithm_info = f"""Algorithm Details (NaVILA Paper):
• Grid: {self.lidar_processor.x_bins}×{self.lidar_processor.y_bins} = {self.lidar_processor.total_bins} cells
• Resolution: {self.lidar_processor.voxel_size_xy*100:.0f}cm per voxel
• Range: X[{self.lidar_processor.range_x[0]:.1f}, {self.lidar_processor.range_x[1]:.1f}], Y[{self.lidar_processor.range_y[0]:.1f}, {self.lidar_processor.range_y[1]:.1f}]m
• Process: Lowest value per voxel + Maximum filter over 5 frames
• Valid points: {len(valid_heights)}/{len(heightmap_flat)} ({100*len(valid_heights)/len(heightmap_flat):.1f}%)"""
        
        fig.text(0.02, 0.02, algorithm_info, fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # 为文本留出空间
        plt.show()
    
    def run_all_tests(self):
        """运行所有测试场景"""
        scenarios = ["flat", "stairs", "terrain", "obstacles"]
        
        for scenario in scenarios:
            print(f"\n{'='*50}")
            print(f"Testing: {scenario.upper()} TERRAIN")
            print(f"{'='*50}")
            
            heightmap, points = self.test_scenario(scenario)
            
            # 显示统计信息
            heightmap_flat = heightmap.cpu().numpy().flatten()
            valid_heights = heightmap_flat[heightmap_flat > -9.0]
            
            print(f"Results:")
            print(f"  Original points: {len(points)}")
            print(f"  Height map cells: {len(heightmap_flat)}")
            print(f"  Valid height cells: {len(valid_heights)}")
            print(f"  Height range: [{np.min(valid_heights):.3f}, {np.max(valid_heights):.3f}] m")
            print(f"  Height std: {np.std(valid_heights):.3f} m")
            
            # 可视化
            self.visualize_results(heightmap, points, scenario)

def main():
    parser = argparse.ArgumentParser(description='Test Height Map Algorithm')
    parser.add_argument('--scenario', type=str, default='all',
                       choices=['flat', 'stairs', 'terrain', 'obstacles', 'all'],
                       help='Test scenario to run')
    
    args = parser.parse_args()
    
    print("Height Map Algorithm Tester")
    print("=" * 50)
    print("Testing the NaVILA paper height map algorithm:")
    print("• Grid: 17×27 (459 dimensions)")
    print("• Resolution: 6cm per voxel")
    print("• Algorithm: Lowest value per voxel + Maximum filter over 5 frames")
    print("• Range: X=[-0.8, 0.2], Y=[-0.8, 0.8], Z=[0, 5] meters")
    print("=" * 50)
    
    tester = HeightMapTester()
    
    if args.scenario == 'all':
        tester.run_all_tests()
    else:
        print(f"\nTesting single scenario: {args.scenario}")
        heightmap, points = tester.test_scenario(args.scenario)
        tester.visualize_results(heightmap, points, args.scenario)

if __name__ == "__main__":
    main()