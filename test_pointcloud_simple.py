#!/usr/bin/env python3
"""
简单点云数据测试脚本
直接接收和显示点云数据，用于调试
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# Import Unitree SDK2 components
try:
    from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
    UNITREE_SDK2_AVAILABLE = True
    print("✓ Unitree SDK2 available")
except ImportError as e:
    UNITREE_SDK2_AVAILABLE = False
    print(f"✗ Unitree SDK2 not available: {e}")
    sys.exit(1)

# Import existing components
sys.path.append('.')
from lidar_height_map_processor import LiDARHeightMapProcessor

class SimplePointCloudTest:
    def __init__(self):
        self.processor = LiDARHeightMapProcessor(device="cpu")
        self.latest_pointcloud = None
        self.latest_heightmap = None
        self.message_count = 0
        self.start_time = time.time()
        
    def pointcloud_callback(self, msg):
        """点云回调函数"""
        try:
            # 转换点云
            points = self.processor.pointcloud2_to_xyz(msg)
            self.latest_pointcloud = points
            self.message_count += 1
            
            if len(points) > 0:
                # 生成heightmap
                self.latest_heightmap = self.processor.process_pointcloud_to_heightmap(msg)
                
                # 每10帧打印一次信息
                if self.message_count % 10 == 1:
                    print(f"Frame #{self.message_count}:")
                    print(f"  Points: {len(points)}")
                    print(f"  HeightMap shape: {self.latest_heightmap.shape}")
                    
                    # 分析点云范围
                    x_range = [points[:, 0].min(), points[:, 0].max()]
                    y_range = [points[:, 1].min(), points[:, 1].max()]
                    z_range = [points[:, 2].min(), points[:, 2].max()]
                    print(f"  X range: [{x_range[0]:.2f}, {x_range[1]:.2f}]")
                    print(f"  Y range: [{y_range[0]:.2f}, {y_range[1]:.2f}]")
                    print(f"  Z range: [{z_range[0]:.2f}, {z_range[1]:.2f}]")
                    
                    # HeightMap统计
                    hm_flat = self.latest_heightmap.cpu().numpy().flatten()
                    valid_heights = hm_flat[hm_flat > -99.0]  # 有效值：不是初始值-100.0
                    print(f"  Valid height cells: {len(valid_heights)}/{len(hm_flat)}")
                    if len(valid_heights) > 0:
                        print(f"  Height range: [{valid_heights.min():.3f}, {valid_heights.max():.3f}]")
                    print()
            else:
                print(f"Frame #{self.message_count}: Empty point cloud")
                
        except Exception as e:
            print(f"Error in callback: {e}")
    
    def init_subscriber(self):
        """初始化订阅者"""
        try:
            print("Initializing DDS channel factory...")
            ChannelFactoryInitialize(0, "eth0")
            
            # 订阅点云数据
            topic = "rt/utlidar/cloud_deskewed"
            self.subscriber = ChannelSubscriber(topic, PointCloud2_)
            self.subscriber.Init(self.pointcloud_callback)
            
            print(f"✓ Subscribed to: {topic}")
            print("Waiting for point cloud data...")
            
        except Exception as e:
            print(f"✗ Failed to initialize subscriber: {e}")
            return False
        
        return True
    
    def save_latest_visualization(self):
        """保存当前数据的可视化"""
        if self.latest_pointcloud is None or self.latest_heightmap is None:
            print("No data to visualize")
            return
        
        print("Creating visualization...")
        
        # 创建图形 - 包含3D子图
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle(f'Point Cloud Test - Frame #{self.message_count}', fontsize=16)
        
        points = self.latest_pointcloud
        heightmap = self.latest_heightmap.cpu().numpy().flatten()
        heightmap_2d = heightmap.reshape(self.processor.y_bins, self.processor.x_bins)
        
        # 显示所有点，不过滤（用于查看完整点云）
        all_points = points
        
        # 现在height map使用所有点的范围，所以大部分点都应该被包含
        mask = (
            (points[:, 0] >= self.processor.range_x[0]) & (points[:, 0] < self.processor.range_x[1]) &
            (points[:, 1] >= self.processor.range_y[0]) & (points[:, 1] < self.processor.range_y[1]) &
            (points[:, 2] >= self.processor.range_z[0]) & (points[:, 2] < self.processor.range_z[1])
        )
        filtered_points = points[mask]
        
        # 创建子图
        ax1 = plt.subplot(2, 3, 1)  # Top view
        ax2 = plt.subplot(2, 3, 2)  # Side view  
        ax3 = plt.subplot(2, 3, 3, projection='3d')  # 3D view
        ax4 = plt.subplot(2, 3, 4)  # Height map
        ax5 = plt.subplot(2, 3, 5, projection='3d')  # Height map 3D
        ax6 = plt.subplot(2, 3, 6)  # Statistics
        
        # 1. 点云顶视图（X-Y平面）
        if len(all_points) > 0:
            scatter1 = ax1.scatter(all_points[:, 0], all_points[:, 1], 
                                  c=all_points[:, 2], cmap='viridis', s=0.1, alpha=0.6)
            ax1.set_title(f'Top View ({len(all_points)} points)')
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_xlim([all_points[:, 0].min()-0.5, all_points[:, 0].max()+0.5])
            ax1.set_ylim([all_points[:, 1].min()-0.5, all_points[:, 1].max()+0.5])
            plt.colorbar(scatter1, ax=ax1, label='Height (m)', shrink=0.8)
        
        # 2. 点云侧视图（X-Z平面）
        if len(all_points) > 0:
            scatter2 = ax2.scatter(all_points[:, 0], all_points[:, 2], 
                                  c=all_points[:, 1], cmap='plasma', s=0.1, alpha=0.6)
            ax2.set_title(f'Side View ({len(all_points)} points)')
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Z (m)')
            ax2.set_xlim([all_points[:, 0].min()-0.5, all_points[:, 0].max()+0.5])
            ax2.set_ylim([all_points[:, 2].min()-0.1, all_points[:, 2].max()+0.1])
            plt.colorbar(scatter2, ax=ax2, label='Y position (m)', shrink=0.8)
        
        # 3. 3D点云视图 - 显示所有点
        if len(all_points) > 0:
            scatter3 = ax3.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2],
                                  c=all_points[:, 2], cmap='viridis', s=0.3, alpha=0.5)
            ax3.set_title(f'3D Point Cloud - ALL {len(all_points)} points')
            ax3.set_xlabel('X (m)')
            ax3.set_ylabel('Y (m)')
            ax3.set_zlabel('Z (m)')
            # 设置相等的轴比例
            ax3.set_box_aspect([1,1,0.5])
        
        # 4. Height Map 2D
        im = ax4.imshow(heightmap_2d, cmap='terrain', origin='lower')
        ax4.set_title('NaVILA Height Map')
        ax4.set_xlabel('X cells')
        ax4.set_ylabel('Y cells')
        plt.colorbar(im, ax=ax4, label='Height (m)', shrink=0.8)
        
        # 5. Height Map 3D表面
        valid_heights = heightmap[heightmap > -99.0]
        if len(valid_heights) > 0:
            x_coords = np.arange(heightmap_2d.shape[1]) * self.processor.voxel_size_xy + self.processor.range_x[0]
            y_coords = np.arange(heightmap_2d.shape[0]) * self.processor.voxel_size_xy + self.processor.range_y[0]
            X, Y = np.meshgrid(x_coords, y_coords)
            
            # 只显示有效的height数据
            Z = heightmap_2d.copy()
            Z[Z <= -99.0] = np.nan  # 隐藏无效数据
            
            surface = ax5.plot_surface(X, Y, Z, cmap='terrain', alpha=0.8, antialiased=True)
            ax5.set_title('Height Map 3D Surface')
            ax5.set_xlabel('X (m)')
            ax5.set_ylabel('Y (m)')
            ax5.set_zlabel('Height (m)')
            ax5.set_box_aspect([1,1,0.3])
        
        # 6. 统计信息
        ax6.axis('off')
        
        if len(valid_heights) > 0:
            stats_text = f"""ALL POINTS STATISTICS:
            
Total LiDAR points: {len(points)}
Points used in height map: {len(filtered_points)} ({100*len(filtered_points)/len(points):.1f}%)

Point cloud full range:
  X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}] m
  Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}] m  
  Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}] m

Height Map (ALL POINTS):
  Grid: {self.processor.x_bins}×{self.processor.y_bins} = {len(heightmap)} cells
  Resolution: {self.processor.voxel_size_xy:.2f}m per cell
  Coverage: X[{self.processor.range_x[0]:.1f}, {self.processor.range_x[1]:.1f}], Y[{self.processor.range_y[0]:.1f}, {self.processor.range_y[1]:.1f}]
  
Valid height cells: {len(valid_heights)} ({100*len(valid_heights)/len(heightmap):.1f}%)
Height range: [{valid_heights.min():.3f}, {valid_heights.max():.3f}] m
Mean height: {valid_heights.mean():.3f} m"""
        else:
            stats_text = "No valid height data found"
        
        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # 保存图片
        filename = f"pointcloud_test_frame_{self.message_count}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {filename}")
    
    def run_test(self, duration=30):
        """运行测试"""
        print(f"Running test for {duration} seconds...")
        
        if not self.init_subscriber():
            return
        
        # 等待第一个数据
        print("Waiting for first point cloud...")
        timeout = 10
        start_wait = time.time()
        
        while self.latest_pointcloud is None:
            if time.time() - start_wait > timeout:
                print("✗ No data received after 10 seconds")
                return
            time.sleep(0.1)
        
        print("✓ First point cloud received!")
        
        # 运行测试
        test_start = time.time()
        last_save = test_start
        
        try:
            while time.time() - test_start < duration:
                current_time = time.time()
                
                # 每10秒保存一次可视化
                if current_time - last_save >= 10:
                    self.save_latest_visualization()
                    last_save = current_time
                
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\n✗ Test stopped by user")
        
        # 保存最终可视化
        self.save_latest_visualization()
        
        elapsed = time.time() - self.start_time
        print(f"\n✓ Test complete!")
        print(f"Duration: {elapsed:.1f}s")
        print(f"Frames received: {self.message_count}")
        if self.message_count > 0:
            print(f"Average rate: {self.message_count/elapsed:.1f} Hz")

def main():
    print("Simple Point Cloud Test")
    print("=" * 30)
    
    tester = SimplePointCloudTest()
    tester.run_test(duration=30)

if __name__ == "__main__":
    main()