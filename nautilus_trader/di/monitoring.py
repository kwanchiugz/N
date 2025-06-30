# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------
"""
Monitoring and health check system for dependency injection container.
"""

import time
import threading
import psutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, TYPE_CHECKING
from enum import Enum
from collections import defaultdict, deque
import asyncio

from nautilus_trader.common.component import Logger

if TYPE_CHECKING:
    from nautilus_trader.di.container import DIContainer


class HealthStatus(str, Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for DI container operations."""
    
    # Service resolution metrics
    resolution_count: int = 0
    total_resolution_time: float = 0.0
    avg_resolution_time: float = 0.0
    max_resolution_time: float = 0.0
    min_resolution_time: float = float('inf')
    
    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_ratio: float = 0.0
    
    # Error metrics
    resolution_errors: int = 0
    validation_errors: int = 0
    
    # System metrics
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # Container metrics
    registered_services: int = 0
    active_singletons: int = 0
    scoped_instances: int = 0
    
    def update_resolution_time(self, duration: float) -> None:
        """Update resolution time metrics."""
        self.resolution_count += 1
        self.total_resolution_time += duration
        self.avg_resolution_time = self.total_resolution_time / self.resolution_count
        self.max_resolution_time = max(self.max_resolution_time, duration)
        self.min_resolution_time = min(self.min_resolution_time, duration)
        
    def update_cache_metrics(self, hit: bool) -> None:
        """Update cache metrics."""
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            
        total = self.cache_hits + self.cache_misses
        self.cache_hit_ratio = self.cache_hits / total if total > 0 else 0.0


class MetricsCollector:
    """
    Collects and aggregates metrics for DI container operations.
    """
    
    def __init__(self, max_history: int = 1000) -> None:
        """
        Initialize metrics collector.
        
        Parameters
        ----------
        max_history : int
            Maximum number of historical data points to keep
        """
        self._metrics = PerformanceMetrics()
        self._logger = Logger(self.__class__.__name__)
        self._lock = threading.RLock()
        
        # Historical data
        self._max_history = max_history
        self._resolution_times: deque = deque(maxlen=max_history)
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._service_usage: Dict[str, int] = defaultdict(int)
        
        # Real-time monitoring
        self._start_time = time.time()
        self._last_collection = time.time()
        
    def record_resolution(self, service_name: str, duration: float, success: bool = True) -> None:
        """
        Record a service resolution operation.
        
        Parameters
        ----------
        service_name : str
            Name of the resolved service
        duration : float
            Resolution time in seconds
        success : bool
            Whether resolution was successful
        """
        with self._lock:
            duration_ms = duration * 1000
            
            if success:
                self._metrics.update_resolution_time(duration_ms)
                self._resolution_times.append(duration_ms)
                self._service_usage[service_name] += 1
            else:
                self._metrics.resolution_errors += 1
                self._error_counts[f"resolution_error_{service_name}"] += 1
                
    def record_cache_hit(self, service_name: str, hit: bool) -> None:
        """Record cache hit/miss."""
        with self._lock:
            self._metrics.update_cache_metrics(hit)
            
    def record_validation_error(self, error_type: str) -> None:
        """Record validation error."""
        with self._lock:
            self._metrics.validation_errors += 1
            self._error_counts[f"validation_error_{error_type}"] += 1
            
    def update_system_metrics(self) -> None:
        """Update system-level metrics."""
        with self._lock:
            # Memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            self._metrics.memory_usage_mb = memory_info.rss / 1024 / 1024
            
            # CPU usage
            self._metrics.cpu_usage_percent = process.cpu_percent()
            
            self._last_collection = time.time()
            
    def update_container_metrics(self, container: "DIContainer") -> None:
        """
        Update container-specific metrics.
        
        Parameters
        ----------
        container : DIContainer
            Container to collect metrics from
        """
        with self._lock:
            self._metrics.registered_services = len(container._services)
            
            # Count active singletons and scoped instances
            singleton_count = 0
            scoped_count = 0
            
            for provider in container._providers.values():
                if hasattr(provider, 'instance') and provider.instance is not None:
                    if provider.descriptor.lifetime.value == "singleton":
                        singleton_count += 1
                        
            self._metrics.active_singletons = singleton_count
            self._metrics.scoped_instances = len(getattr(container, '_scoped_instances', {}))
            
    def get_metrics(self) -> PerformanceMetrics:
        """Get current metrics snapshot."""
        with self._lock:
            return PerformanceMetrics(
                resolution_count=self._metrics.resolution_count,
                total_resolution_time=self._metrics.total_resolution_time,
                avg_resolution_time=self._metrics.avg_resolution_time,
                max_resolution_time=self._metrics.max_resolution_time,
                min_resolution_time=self._metrics.min_resolution_time,
                cache_hits=self._metrics.cache_hits,
                cache_misses=self._metrics.cache_misses,
                cache_hit_ratio=self._metrics.cache_hit_ratio,
                resolution_errors=self._metrics.resolution_errors,
                validation_errors=self._metrics.validation_errors,
                memory_usage_mb=self._metrics.memory_usage_mb,
                cpu_usage_percent=self._metrics.cpu_usage_percent,
                registered_services=self._metrics.registered_services,
                active_singletons=self._metrics.active_singletons,
                scoped_instances=self._metrics.scoped_instances,
            )
            
    def get_top_services(self, limit: int = 10) -> List[tuple]:
        """Get most frequently used services."""
        with self._lock:
            return sorted(self._service_usage.items(), key=lambda x: x[1], reverse=True)[:limit]
            
    def get_error_summary(self) -> Dict[str, int]:
        """Get error summary."""
        with self._lock:
            return dict(self._error_counts)
            
    def get_percentile_resolution_time(self, percentile: float) -> float:
        """Get resolution time percentile."""
        with self._lock:
            if not self._resolution_times:
                return 0.0
                
            sorted_times = sorted(self._resolution_times)
            index = int(len(sorted_times) * percentile / 100)
            return sorted_times[min(index, len(sorted_times) - 1)]
            
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._metrics = PerformanceMetrics()
            self._resolution_times.clear()
            self._error_counts.clear()
            self._service_usage.clear()
            self._start_time = time.time()


class HealthChecker:
    """
    Health check system for DI container and services.
    """
    
    def __init__(self, container: "DIContainer") -> None:
        """
        Initialize health checker.
        
        Parameters
        ----------
        container : DIContainer
            Container to monitor
        """
        self._container = container
        self._logger = Logger(self.__class__.__name__)
        self._health_checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        self._last_results: Dict[str, HealthCheckResult] = {}
        
        # Register built-in health checks
        self._register_builtin_checks()
        
    def register_health_check(self, name: str, check_func: Callable[[], HealthCheckResult]) -> None:
        """
        Register a custom health check.
        
        Parameters
        ----------
        name : str
            Name of the health check
        check_func : Callable
            Function that performs the health check
        """
        self._health_checks[name] = check_func
        self._logger.info(f"Registered health check: {name}")
        
    def run_health_checks(self) -> Dict[str, HealthCheckResult]:
        """
        Run all registered health checks.
        
        Returns
        -------
        Dict[str, HealthCheckResult]
            Results of all health checks
        """
        results = {}
        
        for name, check_func in self._health_checks.items():
            start_time = time.time()
            try:
                result = check_func()
                result.duration_ms = (time.time() - start_time) * 1000
                result.timestamp = time.time()
                results[name] = result
                self._last_results[name] = result
                
            except Exception as e:
                result = HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {e}",
                    duration_ms=(time.time() - start_time) * 1000,
                    timestamp=time.time(),
                    details={"error": str(e)},
                )
                results[name] = result
                self._last_results[name] = result
                self._logger.error(f"Health check '{name}' failed: {e}")
                
        return results
        
    def get_overall_health(self) -> HealthStatus:
        """
        Get overall system health status.
        
        Returns
        -------
        HealthStatus
            Overall health status
        """
        if not self._last_results:
            return HealthStatus.UNKNOWN
            
        statuses = [result.status for result in self._last_results.values()]
        
        if any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in statuses):
            return HealthStatus.DEGRADED
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
            
    def _register_builtin_checks(self) -> None:
        """Register built-in health checks."""
        
        def container_health() -> HealthCheckResult:
            """Check container health."""
            try:
                service_count = len(self._container._services)
                provider_count = len(self._container._providers)
                
                if service_count == 0:
                    return HealthCheckResult(
                        name="container_health",
                        status=HealthStatus.DEGRADED,
                        message="No services registered",
                        duration_ms=0,
                        timestamp=time.time(),
                        details={"service_count": service_count},
                    )
                    
                if service_count != provider_count:
                    return HealthCheckResult(
                        name="container_health",
                        status=HealthStatus.DEGRADED,
                        message="Service/provider count mismatch",
                        duration_ms=0,
                        timestamp=time.time(),
                        details={
                            "service_count": service_count,
                            "provider_count": provider_count,
                        },
                    )
                    
                return HealthCheckResult(
                    name="container_health",
                    status=HealthStatus.HEALTHY,
                    message=f"Container healthy with {service_count} services",
                    duration_ms=0,
                    timestamp=time.time(),
                    details={
                        "service_count": service_count,
                        "provider_count": provider_count,
                    },
                )
                
            except Exception as e:
                return HealthCheckResult(
                    name="container_health",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Container check failed: {e}",
                    duration_ms=0,
                    timestamp=time.time(),
                    details={"error": str(e)},
                )
                
        def memory_health() -> HealthCheckResult:
            """Check memory usage."""
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                # Thresholds (configurable)
                warning_threshold = 500  # MB
                critical_threshold = 1000  # MB
                
                if memory_mb > critical_threshold:
                    status = HealthStatus.UNHEALTHY
                    message = f"High memory usage: {memory_mb:.1f}MB"
                elif memory_mb > warning_threshold:
                    status = HealthStatus.DEGRADED
                    message = f"Elevated memory usage: {memory_mb:.1f}MB"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Memory usage normal: {memory_mb:.1f}MB"
                    
                return HealthCheckResult(
                    name="memory_health",
                    status=status,
                    message=message,
                    duration_ms=0,
                    timestamp=time.time(),
                    details={
                        "memory_mb": memory_mb,
                        "warning_threshold": warning_threshold,
                        "critical_threshold": critical_threshold,
                    },
                )
                
            except Exception as e:
                return HealthCheckResult(
                    name="memory_health",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Memory check failed: {e}",
                    duration_ms=0,
                    timestamp=time.time(),
                    details={"error": str(e)},
                )
                
        def service_resolution_health() -> HealthCheckResult:
            """Check service resolution capability."""
            try:
                # Try to resolve a few core services
                test_services = []
                for interface in list(self._container._services.keys())[:3]:  # Test first 3 services
                    try:
                        instance = self._container.resolve(interface)
                        test_services.append(interface.__name__)
                    except Exception as e:
                        return HealthCheckResult(
                            name="service_resolution_health",
                            status=HealthStatus.UNHEALTHY,
                            message=f"Failed to resolve {interface.__name__}: {e}",
                            duration_ms=0,
                            timestamp=time.time(),
                            details={"failed_service": interface.__name__, "error": str(e)},
                        )
                        
                return HealthCheckResult(
                    name="service_resolution_health",
                    status=HealthStatus.HEALTHY,
                    message=f"Service resolution working, tested {len(test_services)} services",
                    duration_ms=0,
                    timestamp=time.time(),
                    details={"tested_services": test_services},
                )
                
            except Exception as e:
                return HealthCheckResult(
                    name="service_resolution_health",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Service resolution check failed: {e}",
                    duration_ms=0,
                    timestamp=time.time(),
                    details={"error": str(e)},
                )
                
        # Register all built-in checks
        self.register_health_check("container_health", container_health)
        self.register_health_check("memory_health", memory_health)
        self.register_health_check("service_resolution_health", service_resolution_health)


class MonitoringDashboard:
    """
    Simple monitoring dashboard for DI container metrics.
    """
    
    def __init__(self, metrics_collector: MetricsCollector, health_checker: HealthChecker) -> None:
        """
        Initialize monitoring dashboard.
        
        Parameters
        ----------
        metrics_collector : MetricsCollector
            Metrics collector instance
        health_checker : HealthChecker
            Health checker instance
        """
        self._metrics_collector = metrics_collector
        self._health_checker = health_checker
        self._logger = Logger(self.__class__.__name__)
        
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get comprehensive dashboard data.
        
        Returns
        -------
        Dict[str, Any]
            Dashboard data including metrics and health status
        """
        # Update system metrics
        self._metrics_collector.update_system_metrics()
        
        # Get current metrics
        metrics = self._metrics_collector.get_metrics()
        
        # Run health checks
        health_results = self._health_checker.run_health_checks()
        overall_health = self._health_checker.get_overall_health()
        
        # Get additional analytics
        top_services = self._metrics_collector.get_top_services()
        error_summary = self._metrics_collector.get_error_summary()
        
        return {
            "timestamp": time.time(),
            "overall_health": overall_health.value,
            "metrics": {
                "resolution_count": metrics.resolution_count,
                "avg_resolution_time": metrics.avg_resolution_time,
                "max_resolution_time": metrics.max_resolution_time,
                "min_resolution_time": metrics.min_resolution_time,
                "cache_hit_ratio": metrics.cache_hit_ratio,
                "resolution_errors": metrics.resolution_errors,
                "validation_errors": metrics.validation_errors,
                "memory_usage_mb": metrics.memory_usage_mb,
                "cpu_usage_percent": metrics.cpu_usage_percent,
                "registered_services": metrics.registered_services,
                "active_singletons": metrics.active_singletons,
                "scoped_instances": metrics.scoped_instances,
            },
            "health_checks": {
                name: {
                    "status": result.status.value,
                    "message": result.message,
                    "duration_ms": result.duration_ms,
                    "timestamp": result.timestamp,
                    "details": result.details,
                }
                for name, result in health_results.items()
            },
            "analytics": {
                "top_services": top_services,
                "error_summary": error_summary,
                "p95_resolution_time": self._metrics_collector.get_percentile_resolution_time(95),
                "p99_resolution_time": self._metrics_collector.get_percentile_resolution_time(99),
            },
        }
        
    def format_dashboard_text(self) -> str:
        """
        Format dashboard data as text report.
        
        Returns
        -------
        str
            Formatted text dashboard
        """
        data = self.get_dashboard_data()
        
        lines = [
            "DI Container Monitoring Dashboard",
            "=" * 40,
            f"Overall Health: {data['overall_health'].upper()}",
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data['timestamp']))}",
            "",
            "Performance Metrics:",
            f"  Resolution Count: {data['metrics']['resolution_count']}",
            f"  Avg Resolution Time: {data['metrics']['avg_resolution_time']:.2f}ms",
            f"  Max Resolution Time: {data['metrics']['max_resolution_time']:.2f}ms",
            f"  Cache Hit Ratio: {data['metrics']['cache_hit_ratio']:.1%}",
            f"  Resolution Errors: {data['metrics']['resolution_errors']}",
            f"  Memory Usage: {data['metrics']['memory_usage_mb']:.1f}MB",
            f"  Registered Services: {data['metrics']['registered_services']}",
            f"  Active Singletons: {data['metrics']['active_singletons']}",
            "",
            "Health Checks:",
        ]
        
        for name, result in data['health_checks'].items():
            status_symbol = "✅" if result['status'] == "healthy" else "⚠️" if result['status'] == "degraded" else "❌"
            lines.append(f"  {status_symbol} {name}: {result['message']}")
            
        if data['analytics']['top_services']:
            lines.extend([
                "",
                "Top Services:",
            ])
            for service, count in data['analytics']['top_services'][:5]:
                lines.append(f"  {service}: {count} resolutions")
                
        return "\n".join(lines)


# Global monitoring instances
_metrics_collector: Optional[MetricsCollector] = None
_health_checker: Optional[HealthChecker] = None
_dashboard: Optional[MonitoringDashboard] = None


def initialize_monitoring(container: "DIContainer") -> None:
    """Initialize global monitoring system."""
    global _metrics_collector, _health_checker, _dashboard
    
    _metrics_collector = MetricsCollector()
    _health_checker = HealthChecker(container)
    _dashboard = MonitoringDashboard(_metrics_collector, _health_checker)


def get_metrics_collector() -> Optional[MetricsCollector]:
    """Get global metrics collector."""
    return _metrics_collector


def get_health_checker() -> Optional[HealthChecker]:
    """Get global health checker."""
    return _health_checker


def get_dashboard() -> Optional[MonitoringDashboard]:
    """Get global monitoring dashboard."""
    return _dashboard