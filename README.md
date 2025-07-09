# NaVILA Velocity Command 部署说明

## 🎯 系统概述

现在的系统实现了正确的流程：
```
VLM Server → Velocity Commands → Go2 Robot → Print输出 (等待Locomotion Policy集成)
```

## 📁 新文件结构

```
navila_server/
├── navila_velocity_server_v2.py    # 🆕 输出velocity commands的VLM服务器
├── test_velocity_server.py         # 🆕 服务器测试脚本
└── requirements.txt

go2-deploy/
├── go2_velocity_controller.py      # 🆕 接收velocity commands并print的控制器
└── requirements.txt
```

## 🚀 部署步骤

### 1. 启动VLM服务器 (GPU机器)
```bash
cd /hdd/haolan/navila_server
python navila_velocity_server_v2.py
```

### 2. 测试服务器功能
```bash
cd /hdd/haolan/navila_server  
python test_velocity_server.py
```

### 3. 启动Go2控制器 (机器狗)
```bash
# 修改IP地址
cd /hdd/haolan/go2-deploy
# 编辑 go2_velocity_controller.py 中的 SERVER_IP
python go2_velocity_controller.py
```

## 🎯 Velocity Commands输出格式

控制器现在会打印如下格式的velocity commands：

```
============================================================
🎯 VELOCITY COMMAND FROM NAVILA VLM:
   Linear X:  0.2000 m/s
   Angular Z: 0.0000 rad/s  
   Duration:  0.80 seconds
   Action:    move_forward
   From Queue: False
   Queue Remaining: 2
   Episode Step: 5
   Inference Time: 3.245s
   VLM Output: 'move forward 75 cm'
============================================================
```

## 🔧 Velocity命令映射

基于navila_trainer.py的精确逻辑：

| VLM输出 | 离散化 | Linear X | Angular Z | 说明 |
|---------|--------|----------|-----------|------|
| "stop" | - | 0.0 | 0.0 | 停止 |
| "move forward 50 cm" | 2步×25cm | 0.2 | 0.0 | 前进(每步0.8秒) |
| "turn left 30 degree" | 2步×15° | 0.0 | 0.3 | 左转(每步0.8秒) |
| "turn right 45 degree" | 3步×15° | 0.0 | -0.3 | 右转(每步0.8秒) |

## 🔗 集成Locomotion Policy

当你准备集成locomotion policy时，这些velocity commands可以直接作为输入：

```python
# 从ROS话题或文件读取velocity commands
velocity_cmd = {
    "linear_x": 0.2,      # m/s
    "angular_z": 0.0,     # rad/s  
    "duration": 0.8       # seconds
}

# 与Proprioception和Height Map结合
combined_input = {
    "velocity_command": velocity_cmd,
    "proprioception": robot_state,
    "height_map": terrain_data
}

# 通过locomotion policy生成joint positions
joint_positions = locomotion_policy(combined_input)
```

## 📊 关键API端点

### VLM服务器 (端口8888)
- `GET /` - 健康检查
- `POST /reset` - 重置任务指令
- `POST /get_velocity_command` - 获取velocity command
- `GET /status` - 服务器状态
- `GET /velocity_mappings` - 查看动作映射

### 控制器输出
- ROS话题: `/navila/velocity_command` - velocity commands
- ROS话题: `/navila/status` - 系统状态
- 终端输出: 详细的velocity command信息

## ⚡ 性能特点

1. **精确复制navila_trainer.py逻辑**: 完全相同的VLM推理流程
2. **动作队列系统**: 自动分解复杂动作为多个步骤
3. **量化处理**: 25cm前进步长，15°转向步长
4. **容错机制**: 解析失败时安全停止
5. **实时监控**: 详细的性能和状态信息

这个实现现在完全符合你的需求：VLM处理图像 → 输出velocity commands → 打印供locomotion policy使用！