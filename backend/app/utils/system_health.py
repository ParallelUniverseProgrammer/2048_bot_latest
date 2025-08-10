#!/usr/bin/env python3
"""
System Health Monitor with Graceful Degradation
==============================================

This module provides system health monitoring and graceful degradation strategies
to ensure the system continues operating even when components fail or resources
are limited.

Features:
- CPU and memory monitoring
- GPU resource tracking
- Model performance monitoring
- Automatic degradation triggers
- Fallback mode activation
- Resource cleanup strategies
- Health reporting and alerts
"""

import psutil
import torch
import time
import logging
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class DegradationMode(Enum):
    """Available degradation modes"""
    NORMAL = "normal"
    LIGHTWEIGHT = "lightweight"
    MINIMAL = "minimal"
    EMERGENCY = "emergency"

@dataclass
class HealthMetrics:
    """System health metrics"""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_available: float = 0.0
    gpu_memory_percent: float = 0.0
    gpu_memory_total_gb: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_memory_free_gb: float = 0.0
    gpu_utilization: float = 0.0
    disk_usage_percent: float = 0.0
    load_average: float = 0.0
    
    # Performance metrics
    model_inference_time: float = 0.0
    broadcast_success_rate: float = 100.0
    error_rate: float = 0.0
    
    # Timestamps
    timestamp: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class DegradationSettings:
    """Settings for graceful degradation"""
    mode: DegradationMode = DegradationMode.NORMAL
    
    # Performance settings
    max_fps: int = 10
    batch_size: int = 1
    model_precision: str = "float32"
    enable_caching: bool = True
    
    # Feature toggles
    enable_visualization: bool = True
    enable_attention_weights: bool = True
    enable_detailed_logging: bool = True
    enable_performance_metrics: bool = True
    
    # Resource limits
    max_memory_percent: float = 80.0
    max_cpu_percent: float = 80.0
    max_gpu_memory_percent: float = 90.0
    
    # Timeouts
    model_timeout: float = 10.0
    broadcast_timeout: float = 3.0
    
    # Quality settings
    board_update_interval: float = 0.5
    history_retention_steps: int = 1000

class SystemHealthMonitor:
    """System health monitor with graceful degradation"""
    
    def __init__(self, check_interval: float = 5.0):
        self.check_interval = check_interval
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Health state
        self.current_metrics = HealthMetrics()
        self.health_status = HealthStatus.HEALTHY
        self.degradation_settings = DegradationSettings()
        
        # History tracking
        self.metrics_history: List[HealthMetrics] = []
        self.max_history_size = 100
        
        # Alert thresholds
        self.thresholds = {
            HealthStatus.WARNING: {
                'cpu_percent': 70.0,
                'memory_percent': 75.0,
                'gpu_memory_percent': 80.0,
                'error_rate': 5.0,
                'model_inference_time': 2.0
            },
            HealthStatus.DEGRADED: {
                'cpu_percent': 80.0,
                'memory_percent': 85.0,
                'gpu_memory_percent': 90.0,
                'error_rate': 10.0,
                'model_inference_time': 5.0
            },
            HealthStatus.CRITICAL: {
                'cpu_percent': 90.0,
                'memory_percent': 95.0,
                'gpu_memory_percent': 95.0,
                'error_rate': 20.0,
                'model_inference_time': 10.0
            }
        }
        
        # Callbacks for degradation events
        self.degradation_callbacks: List[Callable] = []
        
        # Emergency flags
        self.emergency_mode = False
        self.last_emergency_check = datetime.now()
        
        # Resource tracking
        self.resource_usage_trend = []
        self.performance_trend = []
        
        logger.info("System health monitor initialized")
    
    def start_monitoring(self):
        """Start health monitoring in background thread"""
        if self.is_monitoring:
            logger.warning("Health monitoring already running")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=self.check_interval + 1)
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                self._collect_metrics()
                self._analyze_health()
                self._apply_degradation_if_needed()
                self._cleanup_old_data()
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def _collect_metrics(self):
        """Collect current system metrics"""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # GPU metrics
            gpu_memory_percent = 0.0
            gpu_utilization = 0.0
            gpu_total_gb = 0.0
            gpu_used_gb = 0.0
            gpu_free_gb = 0.0
            
            if torch.cuda.is_available():
                try:
                    # Prefer mem_get_info for real free/total
                    if hasattr(torch.cuda, 'mem_get_info'):
                        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
                        used_bytes = total_bytes - free_bytes
                    else:
                        # Fallback: approximate with allocated bytes
                        total_bytes = torch.cuda.get_device_properties(0).total_memory
                        used_bytes = torch.cuda.memory_allocated()
                        free_bytes = max(0, total_bytes - used_bytes)

                    gpu_memory_percent = (used_bytes / total_bytes) * 100 if total_bytes > 0 else 0.0
                    gpu_total_gb = total_bytes / (1024**3)
                    gpu_used_gb = used_bytes / (1024**3)
                    gpu_free_gb = free_bytes / (1024**3)
                    # Utilization may not be available in torch; keep zero if missing
                    gpu_utilization = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0.0
                except Exception as e:
                    logger.debug(f"GPU metrics collection failed: {e}")
            
            # Disk usage
            disk_usage = psutil.disk_usage('/')
            disk_usage_percent = disk_usage.percent
            
            # Load average (Unix-like systems)
            load_average = 0.0
            try:
                load_average = psutil.getloadavg()[0]
            except (AttributeError, OSError):
                # Windows doesn't have load average
                load_average = cpu_percent / 100.0
            
            # Update metrics
            self.current_metrics = HealthMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available=memory.available / (1024**3),  # GB
                gpu_memory_percent=gpu_memory_percent,
                gpu_memory_total_gb=gpu_total_gb,
                gpu_memory_used_gb=gpu_used_gb,
                gpu_memory_free_gb=gpu_free_gb,
                gpu_utilization=gpu_utilization,
                disk_usage_percent=disk_usage_percent,
                load_average=load_average,
                model_inference_time=self.current_metrics.model_inference_time,
                broadcast_success_rate=self.current_metrics.broadcast_success_rate,
                error_rate=self.current_metrics.error_rate,
                timestamp=datetime.now(),
                last_updated=datetime.now()
            )
            
            # Add to history
            self.metrics_history.append(self.current_metrics)
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
    
    def _analyze_health(self):
        """Analyze current health status"""
        old_status = self.health_status
        
        # Check against thresholds
        if self._check_threshold(HealthStatus.CRITICAL):
            self.health_status = HealthStatus.CRITICAL
        elif self._check_threshold(HealthStatus.DEGRADED):
            self.health_status = HealthStatus.DEGRADED
        elif self._check_threshold(HealthStatus.WARNING):
            self.health_status = HealthStatus.WARNING
        else:
            self.health_status = HealthStatus.HEALTHY
        
        # Check for emergency conditions
        if self._check_emergency_conditions():
            self.health_status = HealthStatus.EMERGENCY
            self.emergency_mode = True
            logger.critical("EMERGENCY MODE ACTIVATED")
        
        # Log status changes
        if old_status != self.health_status:
            logger.warning(f"Health status changed: {old_status.value} -> {self.health_status.value}")
            self._log_health_change(old_status, self.health_status)
    
    def _check_threshold(self, threshold_level: HealthStatus) -> bool:
        """Check if metrics exceed threshold for given level"""
        if threshold_level not in self.thresholds:
            return False
        
        thresholds = self.thresholds[threshold_level]
        metrics = self.current_metrics
        
        # Check individual thresholds
        checks = [
            metrics.cpu_percent >= thresholds.get('cpu_percent', 100),
            metrics.memory_percent >= thresholds.get('memory_percent', 100),
            metrics.gpu_memory_percent >= thresholds.get('gpu_memory_percent', 100),
            metrics.error_rate >= thresholds.get('error_rate', 100),
            metrics.model_inference_time >= thresholds.get('model_inference_time', 100)
        ]
        
        # Return true if any threshold is exceeded
        return any(checks)
    
    def _check_emergency_conditions(self) -> bool:
        """Check for emergency conditions that require immediate action"""
        metrics = self.current_metrics
        
        emergency_conditions = [
            metrics.memory_percent >= 98.0,
            metrics.cpu_percent >= 98.0,
            metrics.gpu_memory_percent >= 98.0,
            metrics.error_rate >= 50.0,
            metrics.model_inference_time >= 30.0,
            metrics.broadcast_success_rate <= 10.0
        ]
        
        return any(emergency_conditions)
    
    def _apply_degradation_if_needed(self):
        """Apply degradation settings based on health status"""
        old_mode = self.degradation_settings.mode
        
        if self.health_status == HealthStatus.EMERGENCY:
            self._apply_emergency_mode()
        elif self.health_status == HealthStatus.CRITICAL:
            self._apply_critical_mode()
        elif self.health_status == HealthStatus.DEGRADED:
            self._apply_degraded_mode()
        elif self.health_status == HealthStatus.WARNING:
            self._apply_warning_mode()
        else:
            self._apply_normal_mode()
        
        # Notify callbacks if mode changed
        if old_mode != self.degradation_settings.mode:
            self._notify_degradation_callbacks()
    
    def _apply_emergency_mode(self):
        """Apply emergency degradation settings"""
        self.degradation_settings = DegradationSettings(
            mode=DegradationMode.EMERGENCY,
            max_fps=1,
            batch_size=1,
            model_precision="float16",
            enable_caching=False,
            enable_visualization=False,
            enable_attention_weights=False,
            enable_detailed_logging=False,
            enable_performance_metrics=False,
            max_memory_percent=95.0,
            max_cpu_percent=95.0,
            max_gpu_memory_percent=98.0,
            model_timeout=30.0,
            broadcast_timeout=10.0,
            board_update_interval=2.0,
            history_retention_steps=100
        )
        logger.critical("EMERGENCY MODE: Minimal functionality enabled")
    
    def _apply_critical_mode(self):
        """Apply critical degradation settings"""
        self.degradation_settings = DegradationSettings(
            mode=DegradationMode.MINIMAL,
            max_fps=2,
            batch_size=1,
            model_precision="float16",
            enable_caching=True,
            enable_visualization=False,
            enable_attention_weights=False,
            enable_detailed_logging=False,
            enable_performance_metrics=True,
            max_memory_percent=90.0,
            max_cpu_percent=90.0,
            max_gpu_memory_percent=95.0,
            model_timeout=15.0,
            broadcast_timeout=5.0,
            board_update_interval=1.0,
            history_retention_steps=500
        )
        logger.warning("CRITICAL MODE: Reduced functionality")
    
    def _apply_degraded_mode(self):
        """Apply degraded mode settings"""
        self.degradation_settings = DegradationSettings(
            mode=DegradationMode.LIGHTWEIGHT,
            max_fps=5,
            batch_size=1,
            model_precision="float32",
            enable_caching=True,
            enable_visualization=True,
            enable_attention_weights=False,
            enable_detailed_logging=False,
            enable_performance_metrics=True,
            max_memory_percent=85.0,
            max_cpu_percent=85.0,
            max_gpu_memory_percent=90.0,
            model_timeout=10.0,
            broadcast_timeout=3.0,
            board_update_interval=0.75,
            history_retention_steps=750
        )
        logger.info("DEGRADED MODE: Lightweight operation")
    
    def _apply_warning_mode(self):
        """Apply warning mode settings"""
        self.degradation_settings = DegradationSettings(
            mode=DegradationMode.NORMAL,
            max_fps=8,
            batch_size=1,
            model_precision="float32",
            enable_caching=True,
            enable_visualization=True,
            enable_attention_weights=True,
            enable_detailed_logging=True,
            enable_performance_metrics=True,
            max_memory_percent=80.0,
            max_cpu_percent=80.0,
            max_gpu_memory_percent=85.0,
            model_timeout=5.0,
            broadcast_timeout=3.0,
            board_update_interval=0.6,
            history_retention_steps=900
        )
        logger.info("WARNING MODE: Cautious operation")
    
    def _apply_normal_mode(self):
        """Apply normal mode settings"""
        self.degradation_settings = DegradationSettings(
            mode=DegradationMode.NORMAL,
            max_fps=10,
            batch_size=1,
            model_precision="float32",
            enable_caching=True,
            enable_visualization=True,
            enable_attention_weights=True,
            enable_detailed_logging=True,
            enable_performance_metrics=True,
            max_memory_percent=75.0,
            max_cpu_percent=75.0,
            max_gpu_memory_percent=80.0,
            model_timeout=5.0,
            broadcast_timeout=3.0,
            board_update_interval=0.5,
            history_retention_steps=1000
        )
    
    def _notify_degradation_callbacks(self):
        """Notify registered callbacks about degradation changes"""
        for callback in self.degradation_callbacks:
            try:
                callback(self.degradation_settings)
            except Exception as e:
                logger.error(f"Error in degradation callback: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old metrics data"""
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]
    
    def _log_health_change(self, old_status: HealthStatus, new_status: HealthStatus):
        """Log health status change with details"""
        logger.info(f"Health status change: {old_status.value} -> {new_status.value}")
        logger.info(f"Current metrics: CPU={self.current_metrics.cpu_percent:.1f}%, "
                   f"Memory={self.current_metrics.memory_percent:.1f}%, "
                   f"GPU={self.current_metrics.gpu_memory_percent:.1f}%")
    
    def register_degradation_callback(self, callback: Callable):
        """Register callback for degradation events"""
        self.degradation_callbacks.append(callback)
    
    def unregister_degradation_callback(self, callback: Callable):
        """Unregister degradation callback"""
        if callback in self.degradation_callbacks:
            self.degradation_callbacks.remove(callback)
    
    def update_performance_metrics(self, inference_time: float, error_rate: float, 
                                 success_rate: float):
        """Update performance metrics from external sources"""
        self.current_metrics.model_inference_time = inference_time
        self.current_metrics.error_rate = error_rate
        self.current_metrics.broadcast_success_rate = success_rate
        self.current_metrics.last_updated = datetime.now()
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report"""
        return {
            'status': self.health_status.value,
            'degradation_mode': self.degradation_settings.mode.value,
            'metrics': {
                'cpu_percent': self.current_metrics.cpu_percent,
                'memory_percent': self.current_metrics.memory_percent,
                'memory_available_gb': self.current_metrics.memory_available,
                'gpu_memory_percent': self.current_metrics.gpu_memory_percent,
                'gpu_memory_total_gb': self.current_metrics.gpu_memory_total_gb,
                'gpu_memory_used_gb': self.current_metrics.gpu_memory_used_gb,
                'gpu_memory_free_gb': self.current_metrics.gpu_memory_free_gb,
                'gpu_utilization': self.current_metrics.gpu_utilization,
                'disk_usage_percent': self.current_metrics.disk_usage_percent,
                'load_average': self.current_metrics.load_average,
                'model_inference_time': self.current_metrics.model_inference_time,
                'broadcast_success_rate': self.current_metrics.broadcast_success_rate,
                'error_rate': self.current_metrics.error_rate
            },
            'degradation_settings': {
                'max_fps': self.degradation_settings.max_fps,
                'enable_visualization': self.degradation_settings.enable_visualization,
                'enable_attention_weights': self.degradation_settings.enable_attention_weights,
                'model_timeout': self.degradation_settings.model_timeout,
                'board_update_interval': self.degradation_settings.board_update_interval
            },
            'emergency_mode': self.emergency_mode,
            'last_updated': self.current_metrics.last_updated.isoformat(),
            'history_size': len(self.metrics_history)
        }
    
    def force_degradation_mode(self, mode: DegradationMode):
        """Force a specific degradation mode (for testing)"""
        old_mode = self.degradation_settings.mode
        
        if mode == DegradationMode.EMERGENCY:
            self._apply_emergency_mode()
        elif mode == DegradationMode.MINIMAL:
            self._apply_critical_mode()
        elif mode == DegradationMode.LIGHTWEIGHT:
            self._apply_degraded_mode()
        else:
            self._apply_normal_mode()
        
        if old_mode != self.degradation_settings.mode:
            self._notify_degradation_callbacks()
        
        logger.info(f"Forced degradation mode: {mode.value}")
    
    def reset_emergency_mode(self):
        """Reset emergency mode (manual recovery)"""
        self.emergency_mode = False
        self.health_status = HealthStatus.HEALTHY
        self._apply_normal_mode()
        logger.info("Emergency mode reset")


# Global health monitor instance
_health_monitor: Optional[SystemHealthMonitor] = None

def get_health_monitor() -> SystemHealthMonitor:
    """Get global health monitor instance"""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = SystemHealthMonitor()
    return _health_monitor

def initialize_health_monitoring(auto_start: bool = True) -> SystemHealthMonitor:
    """Initialize and optionally start health monitoring"""
    monitor = get_health_monitor()
    if auto_start:
        monitor.start_monitoring()
    return monitor 