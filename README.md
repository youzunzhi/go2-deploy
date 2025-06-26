# GO2-Deploy

GO2机器人运动控制策略部署系统 - 基于Extreme-Parkour-Onboard项目

## 项目概述

本项目是从Extreme-Parkour-Onboard项目迁移而来的GO2机器人运动控制策略部署系统。系统采用模块化设计，支持多种运行模式，具备完整的日志记录、性能监控和部署自动化功能。

## 运动模式切换API

### 运动模式切换机制 (Go2 1.1.7+)

本项目使用新的`motion_switcher` API来控制机器人的运动模式：

#### API参数

| 功能 | API ID | 参数 |
|------|--------|------|
| 关闭运动模式 | 1003 | `{}` |
| 开启运动模式 | 1002 | `{"name": "mcf"}` |

#### 代码示例

```python
def publish_motion_switcher(self, mode: int):
    """发布运动模式切换命令（Go2 1.1.7+）"""
    msg = Request()
    msg.header.identity.id = 0
    msg.header.lease.id = 0
    msg.header.policy.priority = 0
    msg.header.policy.noreply = False
    
    if mode == 0:
        # 释放模式（切换到普通模式）
        msg.header.identity.api_id = 1003
        msg.parameter = '{}'
    elif mode == 1:
        # 选择MCF模式（运动模式）
        msg.header.identity.api_id = 1002
        msg.parameter = '{"name": "mcf"}'
    
    msg.binary = []
    self.motion_switcher_pub.publish(msg)
```

## 项目结构

```
go2-deploy/
├── constants.py          # 常量定义（包含运动模式切换API）
├── config.py            # 配置管理
├── utils.py             # 工具函数
├── robot_controller.py  # 机器人控制器（支持运动模式切换）
├── state_manager.py     # 状态管理器（集成运动模式切换）
├── ros_interface.py     # ROS2接口（支持motion_switcher API）
├── model_manager.py     # 模型管理
├── inference_engine.py  # 推理引擎
├── logger.py            # 日志系统
├── deploy.py            # 部署自动化
├── run_loco_policy.py   # 主运行脚本
└── README.md            # 项目文档
```

## 核心功能

### 1. 运动模式切换

- **运动模式**: 使用手柄直接控制机器人运动
- **站立策略**: 使用AI策略保持机器人站立平衡
- **运动控制策略**: 使用训练好的神经网络进行复杂运动控制
- **模式切换**: 支持实时切换，使用`motion_switcher` API
- **主控模式切换**：
  - `native_sport_mode`：Go2原生运动主模式（通过官方API控制）
  - `stand_policy_mode`：自定义站立主模式
  - `locomotion_policy_mode`：自定义locomotion主模式（训练得到的policy控制）
- **回调注册**：
  - `native_sport_action_callbacks`：Go2原生运动模式下的具体动作（如站立、趴下、平衡等）
  - `main_mode_switch_callbacks`：主控模式切换时的回调（如切换到native_sport_mode/stand_policy_mode/locomotion_policy_mode时的硬件和应用逻辑）

### 2. 安全机制

- **关节限制检查**: 防止关节超出安全范围
- **紧急停止**: 异常情况下自动停止机器人
- **安全检查器**: 实时监控机器人状态

### 3. 性能监控

- **推理时间**: 监控神经网络推理性能
- **控制周期**: 确保稳定的控制频率
- **系统健康**: 监控整体系统状态

### 4. 日志记录

- **系统日志**: 记录运行状态和错误信息
- **性能日志**: 记录性能指标和统计数据
- **健康监控**: 监控系统健康状态

## 安装和配置

### 1. 环境要求

- Python 3.8+
- ROS2 Humble
- PyTorch 1.12+
- CUDA 11.6+ (可选，用于GPU加速)
- Go2软件版本 1.1.7+

### 2. 依赖安装

```bash
# 安装Python依赖
pip install torch torchvision torchaudio
pip install numpy opencv-python
pip install rclpy

# 安装ROS2依赖
sudo apt install ros-humble-unitree-ros2
```

### 3. 模型文件

将训练好的模型文件放置在指定目录：

```
traced/
├── base_jit.pt          # 基础模型
├── vision_weight.pt     # 视觉模型权重
└── config.json          # 模型配置
```

## 使用方法

### 1. 基本运行

```bash
# 使用默认配置运行
python run_loco_policy.py

# 指定模型目录
python run_loco_policy.py --model_dir ./traced

# 使用GPU加速
python run_loco_policy.py --device cuda

# 真实模式（非干运行）
python run_loco_policy.py --nodryrun
```

### 2. 运行模式

```bash
# 运动模式
python run_loco_policy.py --mode sport

# 站立策略
python run_loco_policy.py --mode stand

# 运动控制策略
python run_loco_policy.py --mode locomotion

# 行走模式
python run_loco_policy.py --mode walk
```

### 3. 控制循环模式

```bash
# 定时器模式（推荐）
python run_loco_policy.py --loop_mode timer

# 循环模式
python run_loco_policy.py --loop_mode while
```

### 4. 手柄控制

| 按钮 | 功能 |
|------|------|
| R1 | 站立 |
| R2 | 坐下 |
| X | 平衡站立 |
| L1 | 切换到站立策略 |
| Y | 切换到运动控制策略 |
| L2 | 切换回运动模式 |

## 部署自动化

### 1. 自动部署

```bash
# 运行部署脚本
python deploy.py

# 指定配置
python deploy.py --config deploy_config.yaml
```

### 2. 部署功能

- **环境检查**: 验证系统环境和依赖
- **模型验证**: 检查模型文件完整性
- **服务安装**: 创建systemd服务
- **监控脚本**: 生成监控和重启脚本

### 3. 服务管理

```bash
# 启动服务
sudo systemctl start go2-locomotion

# 停止服务
sudo systemctl stop go2-locomotion

# 查看状态
sudo systemctl status go2-locomotion

# 查看日志
sudo journalctl -u go2-locomotion -f
```

## 配置说明

### 1. 机器人配置

```python
# config.py
class RobotConfiguration:
    # 机器人参数
    num_dof = 12
    num_actions = 12
    
    # 关节限制
    joint_limits_low = [-0.802851, -1.0472, -2.69653, ...]
    joint_limits_high = [0.802851, 4.18879, -0.916298, ...]
    
    # 控制参数
    lin_vel_deadband = 0.1
    ang_vel_deadband = 0.1
```

### 2. 部署配置

```python
# config.py
class DeploymentConfig:
    # 设备配置
    device = "cuda"
    dryrun = True
    
    # 控制参数
    duration = 0.02
    safety_ratio = 1.1
    
    # 日志配置
    logdir = "./logs"
    log_level = "INFO"
```

## 故障排除

### 1. 常见问题

**Q: 运动模式无法切换？**
A: 确保Go2软件版本为1.1.7+，检查ROS2连接状态

**Q: 推理速度慢？**
A: 检查是否使用GPU，调整模型预热参数

**Q: 控制不稳定？**
A: 检查控制周期设置，确保系统负载正常

### 2. 日志分析

```bash
# 查看系统日志
tail -f logs/system.log

# 查看性能日志
tail -f logs/performance.log

# 查看错误日志
grep "ERROR" logs/system.log
```

### 3. 性能调优

- **控制周期**: 根据硬件性能调整`duration`参数
- **模型预热**: 增加预热次数提高推理稳定性
- **安全检查**: 调整`safety_ratio`参数

## 开发指南

### 1. 添加新功能

1. 在相应模块中添加功能
2. 更新配置和常量
3. 添加测试用例
4. 更新文档

### 2. 调试模式

```bash
# 启用调试日志
python run_loco_policy.py --log_level DEBUG

# 使用模拟接口
python run_loco_policy.py --dryrun
```

### 3. 测试

```bash
# 运行单元测试
python -m pytest tests/

# 运行集成测试
python tests/test_integration.py
```

## 版本历史

### v1.0.0 (当前版本)
- 完成基础功能迁移
- 实现运动模式切换API
- 添加完整的日志和监控系统
- 支持部署自动化

### 计划功能
- 支持更多机器人型号
- 添加Web界面
- 实现远程监控
- 支持多机器人集群

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 许可证

本项目基于MIT许可证开源。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue
- 发送邮件
- 参与讨论

---

**注意**: 使用本系统前请确保了解机器人安全操作规范，建议在安全环境下进行测试。确保Go2软件版本为1.1.7或更高。 