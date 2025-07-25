"""
LiDAR Height Map处理模块
基于Unitree L1 LiDAR数据生成legged-loco兼容的height map
"""

import numpy as np
import torch
from collections import deque
from scipy.ndimage import maximum_filter
import time
import struct

class LiDARHeightMapProcessor:
    """
    基于Unitree L1 LiDAR点云数据生成height map的处理器
    遵循legged-loco论文的算法：
    1. 创建2.5D voxel grid height map
    2. 对每个voxel grid选择最低值
    3. 对最近5帧LiDAR点云应用maximum filter进行平滑
    """
    
    def __init__(self, device="cpu"):
        self.device = device
        
        # Height map参数 (基于legged-loco配置)
        self.voxel_size_xy = 0.06  # 6cm分辨率
        self.range_x = [-0.8, 0.2]  # 1m前向范围
        self.range_y = [-0.8, 0.8]  # 1.6m横向范围
        self.range_z = [0.0, 5.0]   # 5m垂直范围
        
        # 计算grid尺寸
        self.x_bins = int((self.range_x[1] - self.range_x[0]) / self.voxel_size_xy)  # 17
        self.y_bins = int((self.range_y[1] - self.range_y[0]) / self.voxel_size_xy)  # 27
        self.total_bins = self.x_bins * self.y_bins  # 459
        
        # 保持最近5帧的height map用于maximum filter
        self.height_map_history = deque(maxlen=5)
        
        print(f"LiDAR Height Map Processor initialized:")
        print(f"  Grid size: {self.x_bins} x {self.y_bins} = {self.total_bins}")
        print(f"  Range X: {self.range_x}, Y: {self.range_y}, Z: {self.range_z}")
        print(f"  Voxel size: {self.voxel_size_xy}m")
    
    def pointcloud2_to_xyz(self, pointcloud_msg):
        """
        将PointCloud2消息转换为XYZ坐标数组
        
        Args:
            pointcloud_msg: PointCloud2_消息
        
        Returns:
            numpy array: shape (N, 3) 的点云坐标
        """
        # 解析PointCloud2数据格式
        # 假设点云格式为XYZ (每个点12字节: float32 x, y, z)
        point_step = pointcloud_msg.point_step
        row_step = pointcloud_msg.row_step
        data = bytes(pointcloud_msg.data)
        
        # 提取XYZ坐标
        points = []
        for i in range(0, len(data), point_step):
            if i + 12 <= len(data):  # 确保有足够的数据
                x, y, z = struct.unpack('fff', data[i:i+12])
                # 过滤无效点
                if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                    points.append([x, y, z])
        
        return np.array(points) if points else np.empty((0, 3))
    
    def process_unitree_heightmap(self, heightmap_msg):
        """
        处理Unitree已预处理的HeightMap消息
        
        Args:
            heightmap_msg: HeightMap_消息
        
        Returns:
            torch.Tensor: shape (1, 459) 的height map tensor
        """
        try:
            # 直接使用Unitree提供的height map数据
            height_data = np.array(heightmap_msg.data, dtype=np.float32)
            
            # 检查数据尺寸是否匹配
            expected_size = heightmap_msg.width * heightmap_msg.height
            if len(height_data) != expected_size:
                print(f"Warning: Height map size mismatch. Expected {expected_size}, got {len(height_data)}")
                # 如果尺寸不匹配，尝试调整
                if len(height_data) > expected_size:
                    height_data = height_data[:expected_size]
                else:
                    # 填充零值
                    padded_data = np.zeros(expected_size, dtype=np.float32)
                    padded_data[:len(height_data)] = height_data
                    height_data = padded_data
            
            # 重塑为2D grid
            height_map_2d = height_data.reshape(heightmap_msg.height, heightmap_msg.width)
            
            # 如果尺寸与legged-loco期望不匹配，进行调整
            if height_map_2d.shape != (self.y_bins, self.x_bins):
                print(f"Resizing height map from {height_map_2d.shape} to ({self.y_bins}, {self.x_bins})")
                from scipy import ndimage
                height_map_2d = ndimage.zoom(height_map_2d, 
                                           (self.y_bins / height_map_2d.shape[0], 
                                            self.x_bins / height_map_2d.shape[1]), 
                                           order=1)  # 双线性插值
            
            # 应用maximum filter平滑
            self.height_map_history.append(height_map_2d)
            if len(self.height_map_history) >= 1:
                # 对历史height map应用maximum filter
                smoothed_map = np.maximum.reduce(list(self.height_map_history))
                
                # 应用3x3 max pooling (如训练时)
                smoothed_map = maximum_filter(smoothed_map, size=3)
                
                # 展平并转换为tensor
                height_map_flat = smoothed_map.flatten()
                
                # 确保维度正确
                if len(height_map_flat) != self.total_bins:
                    # 调整到预期尺寸
                    if len(height_map_flat) > self.total_bins:
                        height_map_flat = height_map_flat[:self.total_bins]
                    else:
                        padded_map = np.zeros(self.total_bins, dtype=np.float32)
                        padded_map[:len(height_map_flat)] = height_map_flat
                        height_map_flat = padded_map
                
                return torch.tensor(height_map_flat, device=self.device, dtype=torch.float32).unsqueeze(0)
            
            return torch.zeros(1, self.total_bins, device=self.device, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error processing Unitree height map: {e}")
            return torch.zeros(1, self.total_bins, device=self.device, dtype=torch.float32)
    
    def process_pointcloud_to_heightmap(self, pointcloud_msg):
        """
        从点云数据生成height map (备用方案)
        
        Args:
            pointcloud_msg: PointCloud2_消息
        
        Returns:
            torch.Tensor: shape (1, 459) 的height map tensor
        """
        try:
            # 转换点云数据
            points = self.pointcloud2_to_xyz(pointcloud_msg)
            
            if len(points) == 0:
                return torch.zeros(1, self.total_bins, device=self.device, dtype=torch.float32)
            
            # 过滤点云到指定范围
            mask = (
                (points[:, 0] >= self.range_x[0]) & (points[:, 0] < self.range_x[1]) &
                (points[:, 1] >= self.range_y[0]) & (points[:, 1] < self.range_y[1]) &
                (points[:, 2] >= self.range_z[0]) & (points[:, 2] < self.range_z[1])
            )
            filtered_points = points[mask]
            
            # 创建height map grid
            height_map = np.full((self.y_bins, self.x_bins), -10.0, dtype=np.float32)
            
            if len(filtered_points) > 0:
                # 计算voxel索引
                x_idx = ((filtered_points[:, 0] - self.range_x[0]) / self.voxel_size_xy).astype(int)
                y_idx = ((filtered_points[:, 1] - self.range_y[0]) / self.voxel_size_xy).astype(int)
                
                # 对每个voxel，选择最低z值 (论文要求)
                for i in range(len(filtered_points)):
                    xi, yi = x_idx[i], y_idx[i]
                    if 0 <= xi < self.x_bins and 0 <= yi < self.y_bins:
                        height_map[yi, xi] = min(height_map[yi, xi], filtered_points[i, 2])
            
            # 应用maximum filter到最近5帧
            self.height_map_history.append(height_map)
            if len(self.height_map_history) >= 1:
                final_height_map = np.maximum.reduce(list(self.height_map_history))
                
                # 应用3x3 max pooling
                smoothed_map = maximum_filter(final_height_map, size=3)
                
                # 展平并转换为tensor
                return torch.tensor(smoothed_map.flatten(), device=self.device, dtype=torch.float32).unsqueeze(0)
            
            return torch.zeros(1, self.total_bins, device=self.device, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error processing point cloud to height map: {e}")
            return torch.zeros(1, self.total_bins, device=self.device, dtype=torch.float32)
    
    def get_current_heightmap_tensor(self):
        """
        获取当前的height map tensor (如果没有新数据则返回零tensor)
        
        Returns:
            torch.Tensor: shape (1, 459) 的height map tensor
        """
        if len(self.height_map_history) > 0:
            # 使用最近的height map
            latest_map = self.height_map_history[-1]
            smoothed_map = maximum_filter(latest_map, size=3)
            return torch.tensor(smoothed_map.flatten(), device=self.device, dtype=torch.float32).unsqueeze(0)
        else:
            return torch.zeros(1, self.total_bins, device=self.device, dtype=torch.float32)