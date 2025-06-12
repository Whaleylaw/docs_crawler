"""
System Monitoring and Logging Module
Tracks application performance, metrics, and provides monitoring dashboards
"""

import psutil
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import deque, defaultdict
import json
import asyncio
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used: float  # in MB
    memory_total: float  # in MB
    disk_percent: float
    disk_used: float  # in GB
    disk_total: float  # in GB
    network_sent: int  # bytes
    network_recv: int  # bytes
    active_connections: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used': self.memory_used,
            'memory_total': self.memory_total,
            'disk_percent': self.disk_percent,
            'disk_used': self.disk_used,
            'disk_total': self.disk_total,
            'network_sent': self.network_sent,
            'network_recv': self.network_recv,
            'active_connections': self.active_connections
        }


@dataclass
class ApplicationMetrics:
    """Application-specific metrics"""
    timestamp: datetime
    active_crawl_jobs: int
    total_documents: int
    total_embeddings: int
    search_requests_per_minute: int
    api_response_time: float
    database_query_time: float
    error_count: int
    warning_count: int
    cache_hit_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'active_crawl_jobs': self.active_crawl_jobs,
            'total_documents': self.total_documents,
            'total_embeddings': self.total_embeddings,
            'search_requests_per_minute': self.search_requests_per_minute,
            'api_response_time': self.api_response_time,
            'database_query_time': self.database_query_time,
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'cache_hit_rate': self.cache_hit_rate
        }


@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    metric: str
    operator: str  # '>', '<', '>=', '<=', '=='
    threshold: float
    duration_minutes: int
    enabled: bool = True
    
    def check_condition(self, value: float) -> bool:
        """Check if the alert condition is met"""
        if self.operator == '>':
            return value > self.threshold
        elif self.operator == '<':
            return value < self.threshold
        elif self.operator == '>=':
            return value >= self.threshold
        elif self.operator == '<=':
            return value <= self.threshold
        elif self.operator == '==':
            return value == self.threshold
        return False


class LogHandler(logging.Handler):
    """Custom log handler for monitoring system"""
    
    def __init__(self, monitor):
        super().__init__()
        self.monitor = monitor
        
    def emit(self, record):
        """Handle log record"""
        try:
            self.monitor.add_log_entry(record)
        except Exception:
            self.handleError(record)


class SystemMonitor:
    """Main system monitoring class"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.system_metrics: deque = deque(maxlen=max_history)
        self.app_metrics: deque = deque(maxlen=max_history)
        self.log_entries: deque = deque(maxlen=max_history * 5)
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: Dict[str, datetime] = {}
        
        # Performance counters
        self.performance_counters = defaultdict(list)
        
        # Thread control
        self.monitoring_thread = None
        self.monitoring_active = False
        
        # Network baseline
        self.network_baseline = None
        
        # Setup logging
        self.setup_logging()
        
        # Load default alert rules
        self.setup_default_alerts()
    
    def setup_logging(self):
        """Setup logging configuration"""
        # Create custom handler
        handler = LogHandler(self)
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add to root logger
        logging.getLogger().addHandler(handler)
    
    def setup_default_alerts(self):
        """Setup default alert rules"""
        default_rules = [
            AlertRule("High CPU Usage", "cpu_percent", ">", 80.0, 5),
            AlertRule("High Memory Usage", "memory_percent", ">", 85.0, 3),
            AlertRule("Low Disk Space", "disk_percent", ">", 90.0, 10),
            AlertRule("High Error Rate", "error_count", ">", 10.0, 1),
            AlertRule("Slow API Response", "api_response_time", ">", 5.0, 2),
            AlertRule("Database Query Slow", "database_query_time", ">", 3.0, 2),
        ]
        self.alert_rules.extend(default_rules)
    
    def start_monitoring(self):
        """Start the monitoring thread"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logging.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring thread"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logging.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                system_metrics = self.collect_system_metrics()
                self.system_metrics.append(system_metrics)
                
                # Collect application metrics
                app_metrics = self.collect_application_metrics()
                self.app_metrics.append(app_metrics)
                
                # Check alerts
                self.check_alerts(system_metrics, app_metrics)
                
                # Sleep for collection interval
                time.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect system performance metrics"""
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used = memory.used / 1024 / 1024  # MB
        memory_total = memory.total / 1024 / 1024  # MB
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        disk_used = disk.used / 1024 / 1024 / 1024  # GB
        disk_total = disk.total / 1024 / 1024 / 1024  # GB
        
        # Network metrics
        network = psutil.net_io_counters()
        if self.network_baseline is None:
            self.network_baseline = network
            network_sent = 0
            network_recv = 0
        else:
            network_sent = network.bytes_sent - self.network_baseline.bytes_sent
            network_recv = network.bytes_recv - self.network_baseline.bytes_recv
        
        # Connection count
        try:
            connections = len(psutil.net_connections())
        except:
            connections = 0
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used=memory_used,
            memory_total=memory_total,
            disk_percent=disk_percent,
            disk_used=disk_used,
            disk_total=disk_total,
            network_sent=network_sent,
            network_recv=network_recv,
            active_connections=connections
        )
    
    def collect_application_metrics(self) -> ApplicationMetrics:
        """Collect application-specific metrics"""
        
        # Get metrics from session state or defaults
        active_crawl_jobs = len(st.session_state.get('active_jobs', {}))
        total_documents = st.session_state.get('total_documents', 0)
        total_embeddings = st.session_state.get('total_embeddings', 0)
        
        # Calculate search requests per minute
        search_requests = self.performance_counters.get('search_requests', [])
        recent_searches = [t for t in search_requests if t > time.time() - 60]
        search_requests_per_minute = len(recent_searches)
        
        # Get average response times
        api_times = self.performance_counters.get('api_response_times', [])
        avg_api_time = sum(api_times[-10:]) / len(api_times[-10:]) if api_times else 0.0
        
        db_times = self.performance_counters.get('db_query_times', [])
        avg_db_time = sum(db_times[-10:]) / len(db_times[-10:]) if db_times else 0.0
        
        # Count recent errors and warnings
        recent_logs = [log for log in self.log_entries if log.get('timestamp', datetime.min) > datetime.now() - timedelta(minutes=5)]
        error_count = len([log for log in recent_logs if log.get('level') == 'ERROR'])
        warning_count = len([log for log in recent_logs if log.get('level') == 'WARNING'])
        
        # Calculate cache hit rate
        cache_hits = self.performance_counters.get('cache_hits', [])
        cache_misses = self.performance_counters.get('cache_misses', [])
        total_cache_requests = len(cache_hits) + len(cache_misses)
        cache_hit_rate = len(cache_hits) / total_cache_requests if total_cache_requests > 0 else 0.0
        
        return ApplicationMetrics(
            timestamp=datetime.now(),
            active_crawl_jobs=active_crawl_jobs,
            total_documents=total_documents,
            total_embeddings=total_embeddings,
            search_requests_per_minute=search_requests_per_minute,
            api_response_time=avg_api_time,
            database_query_time=avg_db_time,
            error_count=error_count,
            warning_count=warning_count,
            cache_hit_rate=cache_hit_rate
        )
    
    def check_alerts(self, system_metrics: SystemMetrics, app_metrics: ApplicationMetrics):
        """Check alert conditions"""
        
        all_metrics = {**system_metrics.to_dict(), **app_metrics.to_dict()}
        
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
                
            if rule.metric in all_metrics:
                value = all_metrics[rule.metric]
                
                if rule.check_condition(value):
                    # Alert condition met
                    if rule.name not in self.active_alerts:
                        self.active_alerts[rule.name] = datetime.now()
                        self.trigger_alert(rule, value)
                    else:
                        # Check if alert has been active long enough
                        duration = datetime.now() - self.active_alerts[rule.name]
                        if duration.total_seconds() >= rule.duration_minutes * 60:
                            self.escalate_alert(rule, value, duration)
                else:
                    # Alert condition not met, clear if active
                    if rule.name in self.active_alerts:
                        del self.active_alerts[rule.name]
                        self.clear_alert(rule, value)
    
    def trigger_alert(self, rule: AlertRule, value: float):
        """Trigger an alert"""
        message = f"ALERT: {rule.name} - {rule.metric} is {value} (threshold: {rule.threshold})"
        logging.warning(message)
        
        # Store alert in session state for UI display
        if 'active_alerts' not in st.session_state:
            st.session_state.active_alerts = []
        
        alert_info = {
            'name': rule.name,
            'metric': rule.metric,
            'value': value,
            'threshold': rule.threshold,
            'timestamp': datetime.now().isoformat(),
            'severity': 'warning'
        }
        st.session_state.active_alerts.append(alert_info)
    
    def escalate_alert(self, rule: AlertRule, value: float, duration: timedelta):
        """Escalate an alert that has been active too long"""
        message = f"CRITICAL: {rule.name} has been active for {duration} - {rule.metric} is {value}"
        logging.error(message)
    
    def clear_alert(self, rule: AlertRule, value: float):
        """Clear an alert"""
        message = f"RESOLVED: {rule.name} - {rule.metric} is now {value}"
        logging.info(message)
    
    def add_log_entry(self, record: logging.LogRecord):
        """Add a log entry to the monitoring system"""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        self.log_entries.append(log_entry)
    
    def record_performance_metric(self, metric_type: str, value: float):
        """Record a performance metric"""
        current_time = time.time()
        
        # Clean old entries (keep last hour)
        cutoff_time = current_time - 3600  # 1 hour
        self.performance_counters[metric_type] = [
            v for v in self.performance_counters[metric_type] 
            if isinstance(v, (int, float)) or v > cutoff_time
        ]
        
        # Add new entry
        if metric_type in ['search_requests', 'cache_hits', 'cache_misses']:
            self.performance_counters[metric_type].append(current_time)
        else:
            self.performance_counters[metric_type].append(value)
    
    def get_system_metrics_history(self, hours: int = 1) -> List[SystemMetrics]:
        """Get system metrics history"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [m for m in self.system_metrics if m.timestamp > cutoff_time]
    
    def get_application_metrics_history(self, hours: int = 1) -> List[ApplicationMetrics]:
        """Get application metrics history"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [m for m in self.app_metrics if m.timestamp > cutoff_time]
    
    def get_log_entries(self, hours: int = 1, level: Optional[str] = None) -> List[Dict]:
        """Get log entries"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_logs = [
            log for log in self.log_entries 
            if log.get('timestamp', datetime.min) > cutoff_time
        ]
        
        if level:
            filtered_logs = [log for log in filtered_logs if log.get('level') == level]
        
        return sorted(filtered_logs, key=lambda x: x.get('timestamp', datetime.min), reverse=True)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        if not self.system_metrics or not self.app_metrics:
            return {
                'status': 'unknown',
                'score': 0,
                'issues': ['No metrics available']
            }
        
        latest_system = self.system_metrics[-1]
        latest_app = self.app_metrics[-1]
        
        issues = []
        score = 100
        
        # Check system metrics
        if latest_system.cpu_percent > 80:
            issues.append(f"High CPU usage: {latest_system.cpu_percent:.1f}%")
            score -= 20
        
        if latest_system.memory_percent > 85:
            issues.append(f"High memory usage: {latest_system.memory_percent:.1f}%")
            score -= 20
        
        if latest_system.disk_percent > 90:
            issues.append(f"Low disk space: {latest_system.disk_percent:.1f}% used")
            score -= 30
        
        # Check application metrics
        if latest_app.error_count > 5:
            issues.append(f"High error rate: {latest_app.error_count} errors in 5 minutes")
            score -= 25
        
        if latest_app.api_response_time > 3.0:
            issues.append(f"Slow API responses: {latest_app.api_response_time:.2f}s average")
            score -= 15
        
        # Determine status
        if score >= 80:
            status = 'healthy'
        elif score >= 60:
            status = 'warning'
        else:
            status = 'critical'
        
        return {
            'status': status,
            'score': max(0, score),
            'issues': issues,
            'active_alerts': len(self.active_alerts),
            'last_updated': datetime.now().isoformat()
        }
    
    def export_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Export metrics data"""
        system_history = self.get_system_metrics_history(hours)
        app_history = self.get_application_metrics_history(hours)
        log_history = self.get_log_entries(hours)
        
        return {
            'export_time': datetime.now().isoformat(),
            'duration_hours': hours,
            'system_metrics': [m.to_dict() for m in system_history],
            'application_metrics': [m.to_dict() for m in app_history],
            'log_entries': log_history,
            'alert_rules': [
                {
                    'name': rule.name,
                    'metric': rule.metric,
                    'operator': rule.operator,
                    'threshold': rule.threshold,
                    'duration_minutes': rule.duration_minutes,
                    'enabled': rule.enabled
                }
                for rule in self.alert_rules
            ],
            'health_status': self.get_health_status()
        }


# Global monitor instance
_monitor_instance = None


def get_monitor() -> SystemMonitor:
    """Get the global monitor instance"""
    global _monitor_instance
    
    if _monitor_instance is None:
        _monitor_instance = SystemMonitor()
        
        # Auto-start monitoring if not running
        if not _monitor_instance.monitoring_active:
            _monitor_instance.start_monitoring()
    
    return _monitor_instance


def record_search_request():
    """Record a search request for metrics"""
    monitor = get_monitor()
    monitor.record_performance_metric('search_requests', time.time())


def record_api_response_time(response_time: float):
    """Record API response time"""
    monitor = get_monitor()
    monitor.record_performance_metric('api_response_times', response_time)


def record_database_query_time(query_time: float):
    """Record database query time"""
    monitor = get_monitor()
    monitor.record_performance_metric('db_query_times', query_time)


def record_cache_hit():
    """Record a cache hit"""
    monitor = get_monitor()
    monitor.record_performance_metric('cache_hits', time.time())


def record_cache_miss():
    """Record a cache miss"""
    monitor = get_monitor()
    monitor.record_performance_metric('cache_misses', time.time())


# Context managers for timing operations
class time_operation:
    """Context manager for timing operations"""
    
    def __init__(self, operation_type: str):
        self.operation_type = operation_type
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if self.operation_type == 'api':
            record_api_response_time(duration)
        elif self.operation_type == 'database':
            record_database_query_time(duration)
        
        return False