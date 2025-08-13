#!/usr/bin/env python3
"""
LiDAR Point Cloud and Height Map Visualization Script

This script captures LiDAR point clouds from the Unitree L1 sensor and visualizes
the height map processing pipeline following the NaVILA paper algorithm.

Features:
- Real-time point cloud visualization  
- Height map generation and display
- Paper-accurate processing (lowest value per voxel + maximum filter)
- Side-by-side comparison of point cloud and height map
- Save captured data for analysis

Usage:
    python visualize_lidar_heightmap.py
    python visualize_lidar_heightmap.py --save_data
    python visualize_lidar_heightmap.py --offline_file pointcloud_data.npy
"""

import sys
import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import struct
from collections import deque
from scipy.ndimage import maximum_filter

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import existing components
from lidar_height_map_processor import LiDARHeightMapProcessor
from utils.hardware_cfgs import ROS_TOPICS

# Try to import Unitree SDK2 components
try:
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import HeightMap_
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LidarState_
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
    from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher, ChannelFactoryInitialize
    UNITREE_SDK2_AVAILABLE = True
    print("Unitree SDK2 available - will use real-time LiDAR data")
except ImportError as e:
    UNITREE_SDK2_AVAILABLE = False
    HeightMap_ = None
    PointCloud2_ = None
    LidarState_ = None
    print(f"Unitree SDK2 not available: {e}")
    print("Will use offline mode only")

class LiDARVisualizationManager:
    """
    管理LiDAR点云数据的获取和可视化
    """
    
    def __init__(self, save_data=False, offline_file=None):
        self.save_data = save_data
        self.offline_file = offline_file
        
        # 初始化height map处理器
        self.lidar_processor = LiDARHeightMapProcessor(device="cpu")
        
        # 数据缓存
        self.current_pointcloud = None
        self.current_heightmap = None  # NaVILA算法生成的heightmap
        self.pointcloud_history = deque(maxlen=100)  # 保存最近100帧用于分析
        
        # 可视化设置
        self.fig = None
        self.axes = None
        self.animation = None
        
        # 离线模式数据
        if offline_file and os.path.exists(offline_file):
            self.offline_data = np.load(offline_file, allow_pickle=True)
            self.offline_index = 0
            print(f"Loaded offline data with {len(self.offline_data)} frames")
        else:
            self.offline_data = None
            
        # 初始化SDK2订阅者 (如果可用)
        if UNITREE_SDK2_AVAILABLE and not offline_file:
            self._init_lidar_subscribers()
    
    def _init_lidar_subscribers(self):
        """
        初始化Unitree SDK2 LiDAR订阅者 - 直接订阅原始点云数据
        """
        try:
            # 初始化通道工厂 - 使用eth0作为默认网络接口
            print("Initializing DDS channel factory...")
            ChannelFactoryInitialize(0, "eth0")
            
            # 订阅原始点云数据 (不依赖sport mode)
            # 两个选择：
            # - "rt/utlidar/cloud": 激光雷达坐标系的原始点云
            # - "rt/utlidar/cloud_deskewed": 去运动畸变后的世界坐标系点云 (推荐)
            pointcloud_topic = "rt/utlidar/cloud_deskewed"  # 使用去畸变的点云
            self.pointcloud_subscriber = ChannelSubscriber(pointcloud_topic, PointCloud2_)
            self.pointcloud_subscriber.Init(self._pointcloud_callback)
            
            print("LiDAR subscribers initialized successfully")
            print(f"Subscribed to point cloud topic: {pointcloud_topic}")
            print("This works without sport mode dependency!")
            print("Waiting for point cloud data...")
            
        except Exception as e:
            print(f"Failed to initialize LiDAR subscribers: {e}")
            print("Make sure robot is connected and LiDAR is enabled")
            print("Will use offline mode")
    
    def _pointcloud_callback(self, msg):
        """处理原始点云数据"""
        try:
            # 转换点云数据
            points = self.lidar_processor.pointcloud2_to_xyz(msg)
            if len(points) > 0:
                self.current_pointcloud = points
                self.pointcloud_history.append({
                    'points': points.copy(),
                    'timestamp': time.time()
                })
                
                # 使用NaVILA论文算法从原始点云生成height map
                self.current_heightmap = self.lidar_processor.process_pointcloud_to_heightmap(msg)
                
                if self.save_data:
                    self._save_pointcloud_data(points, time.time())
                
                print(f"PointCloud: {len(points)} points -> NaVILA heightmap {self.current_heightmap.shape}")
            else:
                print("Received empty point cloud")
        except Exception as e:
            print(f"Error in point cloud callback: {e}")
    
    
    def _save_pointcloud_data(self, points, timestamp):
        """保存点云数据"""
        if not hasattr(self, 'saved_pointclouds'):
            self.saved_pointclouds = []
        
        self.saved_pointclouds.append({
            'points': points,
            'timestamp': timestamp
        })
    
    
    def setup_visualization(self):
        """设置可视化界面"""
        plt.style.use('dark_background')
        
        # 创建子图 - 2x2布局
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 12))
        self.fig.suptitle('LiDAR Point Cloud and NaVILA Height Map Processing', 
                         fontsize=16, color='white')
        
        # 设置子图标题
        self.axes[0, 0].set_title('Point Cloud (Top View)', color='white')
        self.axes[0, 1].set_title('Point Cloud (Side View)', color='white')
        self.axes[1, 0].set_title('NaVILA Height Map (Paper Algorithm)', color='white')
        self.axes[1, 1].set_title('Height Map Statistics', color='white')
        
        # 设置坐标轴
        for ax in self.axes.flatten():
            ax.grid(True, alpha=0.3)
            ax.tick_params(colors='white')
        
        plt.tight_layout()
    
    def update_visualization(self, frame):
        """更新可视化 (动画回调函数)"""
        try:
            # 获取当前数据
            pointcloud, heightmap = self._get_current_data()
            
            if pointcloud is None and heightmap is None:
                return
            
            # 清除所有子图
            for ax in self.axes.flatten():
                ax.clear()
                ax.grid(True, alpha=0.3)
                ax.tick_params(colors='white')
            
            # 重新设置标题
            self.axes[0, 0].set_title('Point Cloud (Top View)', color='white')
            self.axes[0, 1].set_title('Point Cloud (Side View)', color='white')
            self.axes[1, 0].set_title('NaVILA Height Map (Paper Algorithm)', color='white')
            self.axes[1, 1].set_title('Height Map Statistics', color='white')
            
            # 绘制点云
            if pointcloud is not None and len(pointcloud) > 0:
                self._plot_pointcloud(pointcloud)
            
            # 绘制NaVILA height map
            if heightmap is not None:
                self._plot_heightmap(heightmap)
            
        except Exception as e:
            print(f"Error updating visualization: {e}")
    
    def _get_current_data(self):
        """获取当前数据 (实时或离线)"""
        if self.offline_data is not None:
            # 离线模式
            if self.offline_index < len(self.offline_data):
                data = self.offline_data[self.offline_index]
                self.offline_index += 1
                return data.get('pointcloud'), data.get('heightmap')
            else:
                self.offline_index = 0  # 循环播放
                return None, None
        else:
            # 实时模式
            return self.current_pointcloud, self.current_heightmap
    
    def _plot_pointcloud(self, points):
        """绘制点云数据"""
        # 过滤点云到height map范围
        x_range = self.lidar_processor.range_x
        y_range = self.lidar_processor.range_y
        z_range = self.lidar_processor.range_z
        
        mask = (
            (points[:, 0] >= x_range[0]) & (points[:, 0] < x_range[1]) &
            (points[:, 1] >= y_range[0]) & (points[:, 1] < y_range[1]) &
            (points[:, 2] >= z_range[0]) & (points[:, 2] < z_range[1])
        )
        filtered_points = points[mask]
        
        if len(filtered_points) == 0:
            return
        
        # 顶视图 (X-Y平面)
        scatter1 = self.axes[0, 0].scatter(filtered_points[:, 0], filtered_points[:, 1], 
                                          c=filtered_points[:, 2], cmap='viridis', 
                                          s=1, alpha=0.6)
        self.axes[0, 0].set_xlabel('X (m)', color='white')
        self.axes[0, 0].set_ylabel('Y (m)', color='white')
        self.axes[0, 0].set_xlim(x_range)
        self.axes[0, 0].set_ylim(y_range)
        
        # 侧视图 (X-Z平面)
        scatter2 = self.axes[0, 1].scatter(filtered_points[:, 0], filtered_points[:, 2], 
                                          c=filtered_points[:, 1], cmap='plasma', 
                                          s=1, alpha=0.6)
        self.axes[0, 1].set_xlabel('X (m)', color='white')
        self.axes[0, 1].set_ylabel('Z (m)', color='white')
        self.axes[0, 1].set_xlim(x_range)
        self.axes[0, 1].set_ylim(z_range)
        
        # 添加colorbar
        if hasattr(self.fig, 'colorbar1'):
            self.fig.colorbar1.remove()
        if hasattr(self.fig, 'colorbar2'):
            self.fig.colorbar2.remove()
        
        self.fig.colorbar1 = self.fig.colorbar(scatter1, ax=self.axes[0, 0], shrink=0.6)
        self.fig.colorbar1.set_label('Height (m)', color='white')
        self.fig.colorbar2 = self.fig.colorbar(scatter2, ax=self.axes[0, 1], shrink=0.6)
        self.fig.colorbar2.set_label('Y position (m)', color='white')
    
    def _plot_heightmap(self, heightmap_tensor):
        """绘制height map"""
        # 将tensor转换为numpy并重塑为2D
        heightmap_flat = heightmap_tensor.cpu().numpy().flatten()
        heightmap_2d = heightmap_flat.reshape(self.lidar_processor.y_bins, 
                                            self.lidar_processor.x_bins)
        
        # 绘制height map
        im = self.axes[1, 0].imshow(heightmap_2d, cmap='terrain', 
                                   origin='lower', interpolation='bilinear')
        
        # 设置坐标轴标签
        x_ticks = np.linspace(0, self.lidar_processor.x_bins-1, 5)
        y_ticks = np.linspace(0, self.lidar_processor.y_bins-1, 5)
        x_labels = np.linspace(self.lidar_processor.range_x[0], 
                              self.lidar_processor.range_x[1], 5)
        y_labels = np.linspace(self.lidar_processor.range_y[0], 
                              self.lidar_processor.range_y[1], 5)
        
        self.axes[1, 0].set_xticks(x_ticks)
        self.axes[1, 0].set_yticks(y_ticks)
        self.axes[1, 0].set_xticklabels([f'{x:.1f}' for x in x_labels])
        self.axes[1, 0].set_yticklabels([f'{y:.1f}' for y in y_labels])
        self.axes[1, 0].set_xlabel('X (m)', color='white')
        self.axes[1, 0].set_ylabel('Y (m)', color='white')
        
        # 添加colorbar
        if hasattr(self.fig, 'colorbar_heightmap'):
            self.fig.colorbar_heightmap.remove()
        self.fig.colorbar_heightmap = self.fig.colorbar(im, ax=self.axes[1, 0], shrink=0.6)
        self.fig.colorbar_heightmap.set_label('Height (m)', color='white')
        
        # 绘制统计信息
        self._plot_heightmap_stats(heightmap_flat)
    
    def _plot_heightmap_stats(self, heightmap_flat):
        """绘制height map统计信息"""
        # 过滤有效值 (非-100.0的值)
        valid_heights = heightmap_flat[heightmap_flat > -99.0]
        
        if len(valid_heights) > 0:
            # 直方图
            self.axes[1, 1].hist(valid_heights, bins=30, color='cyan', alpha=0.7, edgecolor='white')
            self.axes[1, 1].axvline(np.mean(valid_heights), color='red', linestyle='--', 
                                   label=f'Mean: {np.mean(valid_heights):.2f}m')
            self.axes[1, 1].axvline(np.median(valid_heights), color='orange', linestyle='--', 
                                   label=f'Median: {np.median(valid_heights):.2f}m')
            
            self.axes[1, 1].set_xlabel('Height (m)', color='white')
            self.axes[1, 1].set_ylabel('Frequency', color='white')
            self.axes[1, 1].legend()
            
            # 添加统计文本
            stats_text = f"""NaVILA Algorithm Statistics:
Valid points: {len(valid_heights)}/{len(heightmap_flat)}
Min: {np.min(valid_heights):.2f}m
Max: {np.max(valid_heights):.2f}m
Std: {np.std(valid_heights):.2f}m
Grid size: {self.lidar_processor.x_bins}×{self.lidar_processor.y_bins}
Resolution: {self.lidar_processor.voxel_size_xy:.2f}m"""
            
            self.axes[1, 1].text(0.02, 0.98, stats_text, transform=self.axes[1, 1].transAxes,
                               verticalalignment='top', fontsize=9, color='white',
                               bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    
    def run_visualization(self):
        """运行可视化"""
        self.setup_visualization()
        
        if self.offline_data is not None:
            # 离线模式 - 使用动画
            self.animation = FuncAnimation(self.fig, self.update_visualization, 
                                         interval=100, blit=False, cache_frame_data=False)
        else:
            # 实时模式 - 定期更新
            self.animation = FuncAnimation(self.fig, self.update_visualization, 
                                         interval=50, blit=False, cache_frame_data=False)  # 20Hz更新
        
        plt.show()
        
    
    def save_collected_data(self, filename="lidar_data.npy"):
        """保存收集的数据"""
        if self.save_data:
            data_to_save = {
                'pointclouds': getattr(self, 'saved_pointclouds', []),
                'heightmaps': getattr(self, 'saved_heightmaps', []),
                'config': {
                    'x_range': self.lidar_processor.range_x,
                    'y_range': self.lidar_processor.range_y,
                    'z_range': self.lidar_processor.range_z,
                    'voxel_size': self.lidar_processor.voxel_size_xy,
                    'grid_size': (self.lidar_processor.x_bins, self.lidar_processor.y_bins)
                }
            }
            
            np.save(filename, data_to_save)
            print(f"Data saved to {filename}")
            print(f"Saved {len(data_to_save['pointclouds'])} point cloud frames")
            print(f"Saved {len(data_to_save['heightmaps'])} height map frames")

def main():
    parser = argparse.ArgumentParser(description='Visualize LiDAR Point Cloud and Height Map')
    parser.add_argument('--save_data', action='store_true', 
                       help='Save captured data for later analysis')
    parser.add_argument('--offline_file', type=str, 
                       help='Load and visualize offline data from .npy file')
    
    args = parser.parse_args()
    
    print("Starting LiDAR Visualization...")
    print("=" * 50)
    print("Features:")
    print("- Real-time point cloud visualization")
    print("- Height map generation following NaVILA paper")
    print("- Paper-accurate algorithm: lowest value per voxel + maximum filter")
    print("- Grid: 17×27 (459 dimensions), Resolution: 6cm")
    print("- Range: X=[-0.8,0.2], Y=[-0.8,0.8], Z=[0,5] meters")
    print("=" * 50)
    
    # 创建可视化管理器
    viz_manager = LiDARVisualizationManager(
        save_data=args.save_data,
        offline_file=args.offline_file
    )
    
    try:
        # 运行可视化
        viz_manager.run_visualization()
        
    except KeyboardInterrupt:
        print("\nVisualization stopped by user")
    
    finally:
        # 保存数据 (如果启用)
        if args.save_data:
            viz_manager.save_collected_data()
        
        print("Visualization ended")

if __name__ == "__main__":
    main()