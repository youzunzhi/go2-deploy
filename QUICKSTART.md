# NaVILA Go2 快速部署指南

## 📁 最终代码结构
```
navila_server/
├── navila_server.py       # VLM推理服务器 (输出velocity commands)
├── test_server.py         # 服务器测试脚本
└── requirements.txt       # 服务器依赖

go2-deploy/
├── go2_controller.py      # Go2机器狗控制器 (接收并打印velocity commands)
├── README.md              # 详细部署说明
├── requirements.txt       # 控制器依赖
└── (其他locomotion相关文件...)
```

## 🚀 3步部署

### 1. 启动VLM服务器 (GPU机器)
```bash
cd /hdd/haolan/navila_server
python navila_server.py
```

### 2. 测试服务器 (可选)
```bash
cd /hdd/haolan/navila_server
python test_server.py
```

### 3. 启动Go2控制器 (机器狗)
```bash
cd /hdd/haolan/go2-deploy
# 修改 go2_controller.py 中的 SERVER_IP = "你的GPU服务器IP"
python go2_controller.py
```

## 📤 输出示例
```
============================================================
🎯 VELOCITY COMMAND FROM NAVILA VLM:
   Linear X:  0.2000 m/s
   Angular Z: 0.0000 rad/s  
   Duration:  0.80 seconds
   Action:    move_forward
============================================================
```

## 🔗 后续集成
将打印的velocity commands与Proprioception和Height Map结合，输入locomotion policy即可！

---
**注意**: 以后修改代码时直接编辑这些文件，不要创建新版本！