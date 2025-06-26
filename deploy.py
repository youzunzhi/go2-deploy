#!/usr/bin/env python3
# deploy.py
# 部署脚本

import os
import sys
import argparse
import subprocess
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional


class Deployer:
    """部署器"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化部署器
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file
        self.config = self._load_config()
        
        # 项目根目录
        self.project_root = Path(__file__).parent
        self.deploy_dir = self.project_root / "deploy"
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置"""
        if self.config_file and os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                return json.load(f)
        
        # 默认配置
        return {
            "model_dir": None,
            "log_dir": "./logs",
            "device": "cuda",
            "duration": 0.02,
            "dryrun": True,
            "mode": "locomotion",
            "loop_mode": "timer",
            "ros2_workspace": None,
            "python_path": sys.executable
        }
    
    def setup_environment(self):
        """设置环境"""
        print("设置部署环境...")
        
        # 创建部署目录
        self.deploy_dir.mkdir(exist_ok=True)
        
        # 创建日志目录
        log_dir = Path(self.config["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建配置文件
        self._create_deploy_config()
        
        print("环境设置完成")
    
    def _create_deploy_config(self):
        """创建部署配置"""
        deploy_config = {
            "deployment": {
                "model_dir": self.config["model_dir"],
                "log_dir": str(Path(self.config["log_dir"]).absolute()),
                "device": self.config["device"],
                "duration": self.config["duration"],
                "dryrun": self.config["dryrun"],
                "mode": self.config["mode"],
                "loop_mode": self.config["loop_mode"]
            },
            "ros2": {
                "workspace": self.config["ros2_workspace"],
                "source_script": "source /opt/ros/humble/setup.bash"
            },
            "python": {
                "executable": self.config["python_path"],
                "requirements": [
                    "torch>=1.12.0",
                    "numpy>=1.21.0",
                    "rclpy>=3.0.0"
                ]
            }
        }
        
        config_file = self.deploy_dir / "deploy_config.json"
        with open(config_file, 'w') as f:
            json.dump(deploy_config, f, indent=2, ensure_ascii=False)
        
        print(f"部署配置已保存到: {config_file}")
    
    def install_dependencies(self):
        """安装依赖"""
        print("安装Python依赖...")
        
        requirements = self.config.get("requirements", [
            "torch>=1.12.0",
            "numpy>=1.21.0",
            "rclpy>=3.0.0"
        ])
        
        for req in requirements:
            try:
                subprocess.run([
                    self.config["python_path"], "-m", "pip", "install", req
                ], check=True)
                print(f"已安装: {req}")
            except subprocess.CalledProcessError as e:
                print(f"安装失败: {req}, 错误: {e}")
    
    def validate_model(self):
        """验证模型"""
        model_dir = self.config["model_dir"]
        if not model_dir:
            print("警告: 未配置模型目录")
            return False
        
        model_path = Path(model_dir)
        if not model_path.exists():
            print(f"错误: 模型目录不存在: {model_dir}")
            return False
        
        # 检查必需文件
        required_files = ["base_jit.pt", "vision_weight.pt", "config.json"]
        missing_files = []
        
        for file in required_files:
            if not (model_path / file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"错误: 模型目录缺少必需文件: {missing_files}")
            return False
        
        print(f"模型验证通过: {model_dir}")
        return True
    
    def create_launch_script(self):
        """创建启动脚本"""
        print("创建启动脚本...")
        
        # 创建bash启动脚本
        bash_script = self.deploy_dir / "run.sh"
        with open(bash_script, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# GO2运动控制策略启动脚本\n\n")
            
            # 设置环境
            f.write("set -e\n\n")
            
            # 加载ROS2环境
            if self.config["ros2_workspace"]:
                f.write(f"source {self.config['ros2_workspace']}/install/setup.bash\n")
            else:
                f.write("source /opt/ros/humble/setup.bash\n")
            
            f.write("\n")
            
            # 设置Python路径
            f.write(f"export PYTHONPATH={self.project_root}:$PYTHONPATH\n")
            f.write("\n")
            
            # 构建命令
            cmd_parts = [
                self.config["python_path"],
                str(self.project_root / "run_loco_policy.py")
            ]
            
            if self.config["model_dir"]:
                cmd_parts.extend(["--model_dir", self.config["model_dir"]])
            
            cmd_parts.extend([
                "--device", self.config["device"],
                "--duration", str(self.config["duration"]),
                "--mode", self.config["mode"],
                "--loop_mode", self.config["loop_mode"],
                "--logdir", self.config["log_dir"]
            ])
            
            if not self.config["dryrun"]:
                cmd_parts.append("--nodryrun")
            
            f.write(f"exec {' '.join(cmd_parts)}\n")
        
        # 设置执行权限
        os.chmod(bash_script, 0o755)
        
        print(f"启动脚本已创建: {bash_script}")
    
    def create_systemd_service(self):
        """创建systemd服务"""
        print("创建systemd服务...")
        
        service_content = f"""[Unit]
Description=GO2 Locomotion Policy
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory={self.project_root}
Environment=PYTHONPATH={self.project_root}
ExecStart={self.deploy_dir}/run.sh
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
"""
        
        service_file = self.deploy_dir / "go2-locomotion.service"
        with open(service_file, 'w') as f:
            f.write(service_content)
        
        print(f"systemd服务文件已创建: {service_file}")
        print("要安装服务，请运行:")
        print(f"sudo cp {service_file} /etc/systemd/system/")
        print("sudo systemctl daemon-reload")
        print("sudo systemctl enable go2-locomotion")
        print("sudo systemctl start go2-locomotion")
    
    def create_monitoring_script(self):
        """创建监控脚本"""
        print("创建监控脚本...")
        
        monitor_script = self.deploy_dir / "monitor.py"
        with open(monitor_script, 'w') as f:
            f.write("""#!/usr/bin/env python3
# 监控脚本

import psutil
import time
import json
from pathlib import Path

def get_process_info(process_name):
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
        if process_name in proc.info['name']:
            return proc.info
    return None

def main():
    while True:
        # 检查主进程
        proc_info = get_process_info('run_loco_policy')
        
        if proc_info:
            print(f"进程状态: 运行中 (PID: {proc_info['pid']})")
            print(f"CPU使用率: {proc_info['cpu_percent']:.1f}%")
            print(f"内存使用率: {proc_info['memory_percent']:.1f}%")
        else:
            print("进程状态: 未运行")
        
        # 检查日志文件
        log_dir = Path("./logs")
        if log_dir.exists():
            log_files = list(log_dir.glob("*.log"))
            if log_files:
                latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
                print(f"最新日志: {latest_log}")
        
        print("-" * 50)
        time.sleep(10)

if __name__ == "__main__":
    main()
""")
        
        os.chmod(monitor_script, 0o755)
        print(f"监控脚本已创建: {monitor_script}")
    
    def deploy(self):
        """执行完整部署"""
        print("开始部署GO2运动控制策略...")
        
        # 设置环境
        self.setup_environment()
        
        # 安装依赖
        self.install_dependencies()
        
        # 验证模型
        self.validate_model()
        
        # 创建启动脚本
        self.create_launch_script()
        
        # 创建systemd服务
        self.create_systemd_service()
        
        # 创建监控脚本
        self.create_monitoring_script()
        
        print("\n部署完成!")
        print(f"部署目录: {self.deploy_dir}")
        print("使用以下命令启动:")
        print(f"  {self.deploy_dir}/run.sh")
        print("或使用systemd服务:")
        print("  sudo systemctl start go2-locomotion")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="GO2运动控制策略部署工具")
    
    parser.add_argument(
        "--config", 
        type=str, 
        help="配置文件路径"
    )
    
    parser.add_argument(
        "--model_dir", 
        type=str, 
        help="模型目录路径"
    )
    
    parser.add_argument(
        "--log_dir", 
        type=str, 
        default="./logs",
        help="日志目录路径"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        choices=["cuda", "cpu"],
        help="推理设备"
    )
    
    parser.add_argument(
        "--dryrun", 
        action="store_true",
        default=True,
        help="启用干运行模式"
    )
    
    parser.add_argument(
        "--nodryrun", 
        action="store_true",
        help="禁用干运行模式"
    )
    
    args = parser.parse_args()
    
    # 创建部署器
    deployer = Deployer(args.config)
    
    # 更新配置
    if args.model_dir:
        deployer.config["model_dir"] = args.model_dir
    if args.log_dir:
        deployer.config["log_dir"] = args.log_dir
    if args.device:
        deployer.config["device"] = args.device
    if args.nodryrun:
        deployer.config["dryrun"] = False
    
    # 执行部署
    deployer.deploy()


if __name__ == "__main__":
    main() 