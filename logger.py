# logger.py
# 日志和监控系统

import logging
import os
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
import threading
from collections import deque
import csv


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, max_history: int = 1000):
        """
        初始化性能监控器
        
        Args:
            max_history: 最大历史记录数
        """
        self.max_history = max_history
        self.metrics = {}
        self.history = deque(maxlen=max_history)
        self.lock = threading.Lock()
        
        # 初始化指标
        self._init_metrics()
    
    def _init_metrics(self):
        """初始化指标"""
        self.metrics = {
            "control_cycle_time": {"current": 0.0, "min": float('inf'), "max": 0.0, "avg": 0.0, "count": 0},
            "inference_time": {"current": 0.0, "min": float('inf'), "max": 0.0, "avg": 0.0, "count": 0},
            "proprio_time": {"current": 0.0, "min": float('inf'), "max": 0.0, "avg": 0.0, "count": 0},
            "action_time": {"current": 0.0, "min": float('inf'), "max": 0.0, "avg": 0.0, "count": 0},
            "safety_violations": {"current": 0, "total": 0},
            "mode_changes": {"current": 0, "total": 0},
            "error_count": {"current": 0, "total": 0},
            "cycle_count": {"current": 0, "total": 0}
        }
    
    def update_metric(self, name: str, value: float):
        """更新指标"""
        with self.lock:
            if name in self.metrics:
                metric = self.metrics[name]
                metric["current"] = value
                
                if isinstance(value, (int, float)):
                    metric["min"] = min(metric["min"], value)
                    metric["max"] = max(metric["max"], value)
                    metric["count"] += 1
                    metric["avg"] = (metric["avg"] * (metric["count"] - 1) + value) / metric["count"]
    
    def increment_metric(self, name: str, value: int = 1):
        """增加指标"""
        with self.lock:
            if name in self.metrics:
                metric = self.metrics[name]
                metric["current"] += value
                metric["total"] += value
    
    def record_cycle(self, cycle_data: Dict[str, Any]):
        """记录控制循环数据"""
        with self.lock:
            self.history.append({
                "timestamp": time.time(),
                "data": cycle_data
            })
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取当前指标"""
        with self.lock:
            return self.metrics.copy()
    
    def get_recent_history(self, count: int = 100) -> List[Dict[str, Any]]:
        """获取最近的历史记录"""
        with self.lock:
            return list(self.history)[-count:]
    
    def reset(self):
        """重置监控器"""
        with self.lock:
            self._init_metrics()
            self.history.clear()


class SystemLogger:
    """系统日志器"""
    
    def __init__(self, log_dir: Optional[str] = None, log_level: str = "INFO"):
        """
        初始化系统日志器
        
        Args:
            log_dir: 日志目录
            log_level: 日志级别
        """
        self.log_dir = log_dir
        self.log_level = getattr(logging, log_level.upper())
        
        # 创建日志目录
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
        
        # 设置日志格式
        self.log_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 初始化日志器
        self.logger = logging.getLogger("go2_deploy")
        self.logger.setLevel(self.log_level)
        
        # 清除现有处理器
        self.logger.handlers.clear()
        
        # 添加控制台处理器
        self._add_console_handler()
        
        # 添加文件处理器
        if self.log_dir:
            self._add_file_handler()
        
        # 性能监控器
        self.performance_monitor = PerformanceMonitor()
        
        # 错误记录
        self.error_log = []
        self.error_lock = threading.Lock()
    
    def _add_console_handler(self):
        """添加控制台处理器"""
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(self.log_format)
        self.logger.addHandler(console_handler)
    
    def _add_file_handler(self):
        """添加文件处理器"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"go2_deploy_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(self.log_format)
        self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        """信息日志"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """警告日志"""
        self.logger.warning(message)
    
    def error(self, message: str, exception: Optional[Exception] = None):
        """错误日志"""
        if exception:
            self.logger.error(f"{message}: {exception}", exc_info=True)
        else:
            self.logger.error(message)
        
        # 记录错误
        with self.error_lock:
            self.error_log.append({
                "timestamp": time.time(),
                "message": message,
                "exception": str(exception) if exception else None
            })
    
    def debug(self, message: str):
        """调试日志"""
        self.logger.debug(message)
    
    def log_performance(self, metrics: Dict[str, Any]):
        """记录性能指标"""
        self.performance_monitor.record_cycle(metrics)
        
        # 定期保存性能数据
        if len(self.performance_monitor.history) % 100 == 0:
            self._save_performance_data()
    
    def _save_performance_data(self):
        """保存性能数据"""
        if not self.log_dir:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        perf_file = os.path.join(self.log_dir, f"performance_{timestamp}.csv")
        
        try:
            with open(perf_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # 写入表头
                writer.writerow(['timestamp', 'cycle_time', 'inference_time', 'proprio_time', 'action_time'])
                
                # 写入数据
                for record in self.performance_monitor.get_recent_history():
                    data = record["data"]
                    writer.writerow([
                        record["timestamp"],
                        data.get("cycle_time", 0),
                        data.get("inference_time", 0),
                        data.get("proprio_time", 0),
                        data.get("action_time", 0)
                    ])
            
            self.debug(f"性能数据已保存到: {perf_file}")
            
        except Exception as e:
            self.error(f"保存性能数据失败: {e}")
    
    def save_error_report(self):
        """保存错误报告"""
        if not self.log_dir or not self.error_log:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_file = os.path.join(self.log_dir, f"error_report_{timestamp}.json")
        
        try:
            with open(error_file, 'w') as f:
                json.dump(self.error_log, f, indent=2, ensure_ascii=False)
            
            self.info(f"错误报告已保存到: {error_file}")
            
        except Exception as e:
            self.error(f"保存错误报告失败: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        metrics = self.performance_monitor.get_metrics()
        
        summary = {
            "total_cycles": metrics["cycle_count"]["total"],
            "avg_cycle_time": metrics["control_cycle_time"]["avg"],
            "max_cycle_time": metrics["control_cycle_time"]["max"],
            "avg_inference_time": metrics["inference_time"]["avg"],
            "safety_violations": metrics["safety_violations"]["total"],
            "mode_changes": metrics["mode_changes"]["total"],
            "errors": metrics["error_count"]["total"]
        }
        
        return summary
    
    def print_performance_summary(self):
        """打印性能摘要"""
        summary = self.get_performance_summary()
        
        print("\n=== 性能摘要 ===")
        print(f"总控制循环次数: {summary['total_cycles']}")
        print(f"平均循环时间: {summary['avg_cycle_time']:.5f}s")
        print(f"最大循环时间: {summary['max_cycle_time']:.5f}s")
        print(f"平均推理时间: {summary['avg_inference_time']:.5f}s")
        print(f"安全违规次数: {summary['safety_violations']}")
        print(f"模式切换次数: {summary['mode_changes']}")
        print(f"错误次数: {summary['errors']}")
        print("=" * 20)


class HealthMonitor:
    """健康监控器"""
    
    def __init__(self, logger: SystemLogger):
        """
        初始化健康监控器
        
        Args:
            logger: 系统日志器
        """
        self.logger = logger
        self.health_status = {
            "system": "healthy",
            "robot_controller": "healthy",
            "inference_engine": "healthy",
            "ros_interface": "healthy",
            "safety_system": "healthy"
        }
        
        self.health_checks = {
            "cycle_time_threshold": 0.1,  # 100ms
            "inference_time_threshold": 0.05,  # 50ms
            "safety_violation_threshold": 10,
            "error_threshold": 5
        }
        
        self.last_check_time = time.time()
        self.check_interval = 10.0  # 10秒检查一次
    
    def update_health_status(self, component: str, status: str, details: Optional[str] = None):
        """更新健康状态"""
        self.health_status[component] = status
        
        if status != "healthy":
            self.logger.warning(f"组件 {component} 状态异常: {status}")
            if details:
                self.logger.debug(f"详细信息: {details}")
    
    def check_system_health(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """检查系统健康状态"""
        current_time = time.time()
        
        # 定期检查
        if current_time - self.last_check_time < self.check_interval:
            return self.health_status
        
        self.last_check_time = current_time
        
        # 检查循环时间
        cycle_time = metrics.get("control_cycle_time", {}).get("current", 0)
        if cycle_time > self.health_checks["cycle_time_threshold"]:
            self.update_health_status("system", "degraded", f"循环时间过长: {cycle_time:.3f}s")
        
        # 检查推理时间
        inference_time = metrics.get("inference_time", {}).get("current", 0)
        if inference_time > self.health_checks["inference_time_threshold"]:
            self.update_health_status("inference_engine", "degraded", f"推理时间过长: {inference_time:.3f}s")
        
        # 检查安全违规
        safety_violations = metrics.get("safety_violations", {}).get("total", 0)
        if safety_violations > self.health_checks["safety_violation_threshold"]:
            self.update_health_status("safety_system", "warning", f"安全违规次数过多: {safety_violations}")
        
        # 检查错误次数
        error_count = metrics.get("error_count", {}).get("total", 0)
        if error_count > self.health_checks["error_threshold"]:
            self.update_health_status("system", "error", f"错误次数过多: {error_count}")
        
        return self.health_status
    
    def get_health_summary(self) -> Dict[str, Any]:
        """获取健康摘要"""
        healthy_components = sum(1 for status in self.health_status.values() if status == "healthy")
        total_components = len(self.health_status)
        
        return {
            "overall_health": "healthy" if healthy_components == total_components else "degraded",
            "healthy_components": healthy_components,
            "total_components": total_components,
            "component_status": self.health_status.copy()
        }
    
    def print_health_summary(self):
        """打印健康摘要"""
        summary = self.get_health_summary()
        
        print("\n=== 系统健康状态 ===")
        print(f"整体状态: {summary['overall_health']}")
        print(f"健康组件: {summary['healthy_components']}/{summary['total_components']}")
        
        for component, status in summary['component_status'].items():
            status_icon = "✅" if status == "healthy" else "⚠️" if status == "degraded" else "❌"
            print(f"  {status_icon} {component}: {status}")
        
        print("=" * 20) 