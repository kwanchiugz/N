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
Resilience and graceful degradation mechanisms for dependency injection.
"""

import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, Union
from enum import Enum
from collections import defaultdict

from nautilus_trader.common.component import Logger
from nautilus_trader.di.exceptions import ResolutionError, ConfigurationError


class FailureMode(str, Enum):
    """Service failure modes."""
    FAIL_FAST = "fail_fast"                    # Immediately fail without retry
    GRACEFUL_DEGRADATION = "graceful"          # Try fallback mechanisms
    CIRCUIT_BREAKER = "circuit_breaker"        # Circuit breaker pattern
    RETRY_WITH_BACKOFF = "retry_backoff"       # Retry with exponential backoff
    DEFAULT_INSTANCE = "default_instance"      # Return default/mock instance


class ServiceHealth(str, Enum):
    """Service health states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class FailureRecord:
    """Record of a service failure."""
    timestamp: float
    error_type: str
    error_message: str
    resolution_chain: List[str] = field(default_factory=list)
    recovery_attempted: bool = False
    recovery_successful: bool = False


@dataclass
class ResilienceConfig:
    """Configuration for service resilience."""
    failure_mode: FailureMode = FailureMode.GRACEFUL_DEGRADATION
    max_retries: int = 3
    retry_delay: float = 1.0
    backoff_multiplier: float = 2.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 30.0
    health_check_interval: float = 10.0
    degradation_timeout: float = 300.0  # 5 minutes


class FallbackProvider(ABC):
    """Abstract base for fallback service providers."""
    
    @abstractmethod
    def can_provide_fallback(self, interface: Type) -> bool:
        """Check if fallback can be provided for interface."""
        pass
        
    @abstractmethod
    def create_fallback(self, interface: Type, error: Exception) -> Any:
        """Create fallback instance."""
        pass


class MockFallbackProvider(FallbackProvider):
    """Provides mock instances as fallbacks."""
    
    def __init__(self) -> None:
        self._logger = Logger(self.__class__.__name__)
        
    def can_provide_fallback(self, interface: Type) -> bool:
        """Check if we can create a mock for this interface."""
        # Can mock interfaces/abstract classes, not concrete classes
        return hasattr(interface, '__abstractmethods__') or interface.__name__.startswith('I')
        
    def create_fallback(self, interface: Type, error: Exception) -> Any:
        """Create mock instance."""
        try:
            from unittest.mock import MagicMock
            mock = MagicMock(spec=interface)
            mock.__class__.__name__ = f"Mock{interface.__name__}"
            self._logger.warning(f"Created mock fallback for {interface.__name__} due to: {error}")
            return mock
        except ImportError:
            raise ConfigurationError(
                f"Cannot create mock fallback for {interface.__name__} - unittest.mock not available",
                suggestion="Install unittest.mock or provide custom fallback",
            )


class DefaultInstanceFallbackProvider(FallbackProvider):
    """Provides pre-configured default instances."""
    
    def __init__(self) -> None:
        self._defaults: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable] = {}
        self._logger = Logger(self.__class__.__name__)
        
    def register_default(self, interface: Type, instance: Any) -> None:
        """Register default instance for interface."""
        self._defaults[interface] = instance
        self._logger.info(f"Registered default instance for {interface.__name__}")
        
    def register_default_factory(self, interface: Type, factory: Callable) -> None:
        """Register factory for creating default instances."""
        self._factories[interface] = factory
        self._logger.info(f"Registered default factory for {interface.__name__}")
        
    def can_provide_fallback(self, interface: Type) -> bool:
        """Check if default is available."""
        return interface in self._defaults or interface in self._factories
        
    def create_fallback(self, interface: Type, error: Exception) -> Any:
        """Create default instance."""
        if interface in self._defaults:
            self._logger.warning(f"Using default instance for {interface.__name__} due to: {error}")
            return self._defaults[interface]
            
        if interface in self._factories:
            try:
                instance = self._factories[interface]()
                self._logger.warning(f"Created default instance for {interface.__name__} due to: {error}")
                return instance
            except Exception as factory_error:
                raise ConfigurationError(
                    f"Default factory failed for {interface.__name__}: {factory_error}",
                    suggestion="Check default factory implementation",
                ) from factory_error
                
        raise ConfigurationError(f"No default available for {interface.__name__}")


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 30.0,
        recovery_timeout: float = 5.0,
    ) -> None:
        """
        Initialize circuit breaker.
        
        Parameters
        ----------
        failure_threshold : int
            Number of failures before opening circuit
        timeout : float
            How long to keep circuit open (seconds)
        recovery_timeout : float
            How long to wait before trying to close circuit
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.recovery_timeout = recovery_timeout
        
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._last_success_time = 0.0
        self._state = "closed"  # closed, open, half_open
        self._lock = threading.RLock()
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        
        Parameters
        ----------
        func : Callable
            Function to execute
        *args, **kwargs
            Function arguments
            
        Returns
        -------
        Any
            Function result
            
        Raises
        ------
        Exception
            If circuit is open or function fails
        """
        with self._lock:
            current_time = time.time()
            
            # Check if circuit should transition states
            if self._state == "open":
                if current_time - self._last_failure_time > self.timeout:
                    self._state = "half_open"
                else:
                    raise ResolutionError(
                        f"Circuit breaker is open (failures: {self._failure_count})",
                        original_error=Exception("Circuit breaker open"),
                    )
                    
            try:
                result = func(*args, **kwargs)
                
                # Success - reset or close circuit
                if self._state == "half_open":
                    self._state = "closed"
                    self._failure_count = 0
                    
                self._last_success_time = current_time
                return result
                
            except Exception as e:
                self._failure_count += 1
                self._last_failure_time = current_time
                
                if self._failure_count >= self.failure_threshold:
                    self._state = "open"
                    
                raise e
                
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        with self._lock:
            return {
                "state": self._state,
                "failure_count": self._failure_count,
                "last_failure_time": self._last_failure_time,
                "last_success_time": self._last_success_time,
            }
            
    def reset(self) -> None:
        """Reset circuit breaker."""
        with self._lock:
            self._state = "closed"
            self._failure_count = 0
            self._last_failure_time = 0.0


class ServiceHealthMonitor:
    """Monitors service health and tracks failures."""
    
    def __init__(self, config: ResilienceConfig) -> None:
        """
        Initialize health monitor.
        
        Parameters
        ----------
        config : ResilienceConfig
            Resilience configuration
        """
        self._config = config
        self._logger = Logger(self.__class__.__name__)
        
        # Health tracking
        self._service_health: Dict[Type, ServiceHealth] = defaultdict(lambda: ServiceHealth.HEALTHY)
        self._failure_history: Dict[Type, List[FailureRecord]] = defaultdict(list)
        self._circuit_breakers: Dict[Type, CircuitBreaker] = {}
        self._last_health_check: Dict[Type, float] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
    def record_failure(
        self,
        interface: Type,
        error: Exception,
        resolution_chain: Optional[List[Type]] = None,
    ) -> None:
        """
        Record service failure.
        
        Parameters
        ----------
        interface : Type
            Failed service interface
        error : Exception
            Failure exception
        resolution_chain : List[Type], optional
            Resolution chain when failure occurred
        """
        with self._lock:
            failure = FailureRecord(
                timestamp=time.time(),
                error_type=type(error).__name__,
                error_message=str(error),
                resolution_chain=[t.__name__ for t in resolution_chain or []],
            )
            
            self._failure_history[interface].append(failure)
            
            # Update health status
            self._update_health_status(interface)
            
            self._logger.warning(f"Recorded failure for {interface.__name__}: {error}")
            
    def record_success(self, interface: Type) -> None:
        """
        Record successful service resolution.
        
        Parameters
        ----------
        interface : Type
            Successfully resolved service interface
        """
        with self._lock:
            current_time = time.time()
            self._last_health_check[interface] = current_time
            
            # Improve health status if recovering
            current_health = self._service_health[interface]
            if current_health in [ServiceHealth.RECOVERING, ServiceHealth.DEGRADED]:
                # Check if we should mark as healthy
                recent_failures = self._get_recent_failures(interface, 300)  # Last 5 minutes
                if len(recent_failures) == 0:
                    self._service_health[interface] = ServiceHealth.HEALTHY
                    self._logger.info(f"Service {interface.__name__} recovered to healthy state")
                    
    def get_health_status(self, interface: Type) -> ServiceHealth:
        """Get current health status for service."""
        with self._lock:
            return self._service_health[interface]
            
    def should_attempt_recovery(self, interface: Type) -> bool:
        """Check if recovery should be attempted."""
        with self._lock:
            health = self._service_health[interface]
            
            if health == ServiceHealth.HEALTHY:
                return True
                
            if health == ServiceHealth.FAILED:
                # Check if enough time has passed for recovery attempt
                last_failure = self._get_last_failure(interface)
                if last_failure:
                    time_since_failure = time.time() - last_failure.timestamp
                    return time_since_failure > self._config.degradation_timeout
                    
            return health in [ServiceHealth.DEGRADED, ServiceHealth.RECOVERING]
            
    def get_circuit_breaker(self, interface: Type) -> CircuitBreaker:
        """Get or create circuit breaker for service."""
        with self._lock:
            if interface not in self._circuit_breakers:
                self._circuit_breakers[interface] = CircuitBreaker(
                    failure_threshold=self._config.circuit_breaker_threshold,
                    timeout=self._config.circuit_breaker_timeout,
                )
            return self._circuit_breakers[interface]
            
    def get_failure_summary(self, interface: Type) -> Dict[str, Any]:
        """Get failure summary for service."""
        with self._lock:
            failures = self._failure_history.get(interface, [])
            recent_failures = self._get_recent_failures(interface, 3600)  # Last hour
            
            return {
                "health_status": self._service_health[interface].value,
                "total_failures": len(failures),
                "recent_failures": len(recent_failures),
                "last_failure": self._get_last_failure(interface),
                "circuit_breaker_state": (
                    self._circuit_breakers[interface].get_state()
                    if interface in self._circuit_breakers
                    else None
                ),
            }
            
    def cleanup_old_failures(self, max_age_hours: int = 24) -> int:
        """
        Clean up old failure records.
        
        Parameters
        ----------
        max_age_hours : int
            Maximum age in hours for failure records
            
        Returns
        -------
        int
            Number of records cleaned up
        """
        with self._lock:
            cutoff_time = time.time() - (max_age_hours * 3600)
            cleaned_count = 0
            
            for interface, failures in self._failure_history.items():
                original_count = len(failures)
                self._failure_history[interface] = [
                    f for f in failures if f.timestamp > cutoff_time
                ]
                cleaned_count += original_count - len(self._failure_history[interface])
                
            if cleaned_count > 0:
                self._logger.debug(f"Cleaned up {cleaned_count} old failure records")
                
            return cleaned_count
            
    def _update_health_status(self, interface: Type) -> None:
        """Update health status based on failure history."""
        recent_failures = self._get_recent_failures(interface, 300)  # Last 5 minutes
        failure_count = len(recent_failures)
        
        if failure_count == 0:
            self._service_health[interface] = ServiceHealth.HEALTHY
        elif failure_count <= 2:
            self._service_health[interface] = ServiceHealth.DEGRADED
        elif failure_count <= 5:
            self._service_health[interface] = ServiceHealth.FAILING
        else:
            self._service_health[interface] = ServiceHealth.FAILED
            
    def _get_recent_failures(self, interface: Type, seconds: int) -> List[FailureRecord]:
        """Get failures within specified time window."""
        cutoff_time = time.time() - seconds
        return [
            f for f in self._failure_history.get(interface, [])
            if f.timestamp > cutoff_time
        ]
        
    def _get_last_failure(self, interface: Type) -> Optional[FailureRecord]:
        """Get most recent failure."""
        failures = self._failure_history.get(interface, [])
        return failures[-1] if failures else None


class ResilientServiceProvider:
    """
    Service provider wrapper with resilience capabilities.
    """
    
    def __init__(
        self,
        original_provider: Any,
        config: ResilienceConfig,
        health_monitor: ServiceHealthMonitor,
        fallback_providers: Optional[List[FallbackProvider]] = None,
    ) -> None:
        """
        Initialize resilient service provider.
        
        Parameters
        ----------
        original_provider : Any
            Original service provider
        config : ResilienceConfig
            Resilience configuration
        health_monitor : ServiceHealthMonitor
            Health monitor instance
        fallback_providers : List[FallbackProvider], optional
            Fallback providers
        """
        self.original_provider = original_provider
        self.config = config
        self.health_monitor = health_monitor
        self.fallback_providers = fallback_providers or []
        self._logger = Logger(self.__class__.__name__)
        
    def get(self, container: Any, resolution_chain: Optional[set] = None) -> Any:
        """
        Get service with resilience mechanisms.
        
        Parameters
        ----------
        container : Any
            DI container
        resolution_chain : set, optional
            Current resolution chain
            
        Returns
        -------
        Any
            Service instance
        """
        interface = self.original_provider.descriptor.interface
        
        # Check if we should attempt resolution
        if not self.health_monitor.should_attempt_recovery(interface):
            return self._try_fallback(interface, Exception("Service marked as failed"))
            
        try:
            # Apply resilience pattern based on configuration
            if self.config.failure_mode == FailureMode.CIRCUIT_BREAKER:
                return self._resolve_with_circuit_breaker(container, resolution_chain)
            elif self.config.failure_mode == FailureMode.RETRY_WITH_BACKOFF:
                return self._resolve_with_retry(container, resolution_chain)
            else:
                return self._resolve_with_fallback(container, resolution_chain)
                
        except Exception as e:
            self.health_monitor.record_failure(interface, e, list(resolution_chain or []))
            
            if self.config.failure_mode == FailureMode.FAIL_FAST:
                raise e
            else:
                return self._try_fallback(interface, e)
                
    def _resolve_with_circuit_breaker(self, container: Any, resolution_chain: Optional[set]) -> Any:
        """Resolve using circuit breaker pattern."""
        interface = self.original_provider.descriptor.interface
        circuit_breaker = self.health_monitor.get_circuit_breaker(interface)
        
        def resolve():
            result = self.original_provider.get(container, resolution_chain)
            self.health_monitor.record_success(interface)
            return result
            
        return circuit_breaker.call(resolve)
        
    def _resolve_with_retry(self, container: Any, resolution_chain: Optional[set]) -> Any:
        """Resolve with retry and exponential backoff."""
        interface = self.original_provider.descriptor.interface
        last_exception = None
        delay = self.config.retry_delay
        
        for attempt in range(self.config.max_retries + 1):
            try:
                result = self.original_provider.get(container, resolution_chain)
                self.health_monitor.record_success(interface)
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt < self.config.max_retries:
                    self._logger.warning(
                        f"Service resolution failed (attempt {attempt + 1}/{self.config.max_retries + 1}), "
                        f"retrying in {delay}s: {e}"
                    )
                    time.sleep(delay)
                    delay *= self.config.backoff_multiplier
                else:
                    self._logger.error(f"Service resolution failed after {self.config.max_retries + 1} attempts: {e}")
                    
        raise last_exception
        
    def _resolve_with_fallback(self, container: Any, resolution_chain: Optional[set]) -> Any:
        """Resolve with graceful fallback."""
        interface = self.original_provider.descriptor.interface
        
        try:
            result = self.original_provider.get(container, resolution_chain)
            self.health_monitor.record_success(interface)
            return result
        except Exception as e:
            return self._try_fallback(interface, e)
            
    def _try_fallback(self, interface: Type, original_error: Exception) -> Any:
        """Try fallback providers."""
        for fallback_provider in self.fallback_providers:
            if fallback_provider.can_provide_fallback(interface):
                try:
                    fallback = fallback_provider.create_fallback(interface, original_error)
                    self._logger.warning(f"Using fallback for {interface.__name__}")
                    return fallback
                except Exception as fallback_error:
                    self._logger.error(f"Fallback failed for {interface.__name__}: {fallback_error}")
                    
        # No fallback available - raise original error
        raise original_error


class ResilienceManager:
    """
    Manages resilience configuration and monitoring for the DI container.
    """
    
    def __init__(self, config: Optional[ResilienceConfig] = None) -> None:
        """
        Initialize resilience manager.
        
        Parameters
        ----------
        config : ResilienceConfig, optional
            Resilience configuration
        """
        self.config = config or ResilienceConfig()
        self.health_monitor = ServiceHealthMonitor(self.config)
        self.fallback_providers: List[FallbackProvider] = []
        self._logger = Logger(self.__class__.__name__)
        
        # Add default fallback providers
        self.add_fallback_provider(DefaultInstanceFallbackProvider())
        self.add_fallback_provider(MockFallbackProvider())
        
    def add_fallback_provider(self, provider: FallbackProvider) -> None:
        """Add fallback provider."""
        self.fallback_providers.append(provider)
        self._logger.info(f"Added fallback provider: {type(provider).__name__}")
        
    def wrap_provider(self, provider: Any) -> ResilientServiceProvider:
        """
        Wrap provider with resilience capabilities.
        
        Parameters
        ----------
        provider : Any
            Original provider to wrap
            
        Returns
        -------
        ResilientServiceProvider
            Wrapped provider
        """
        return ResilientServiceProvider(
            original_provider=provider,
            config=self.config,
            health_monitor=self.health_monitor,
            fallback_providers=self.fallback_providers,
        )
        
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        # This would collect health information from all monitored services
        return {
            "config": {
                "failure_mode": self.config.failure_mode.value,
                "max_retries": self.config.max_retries,
                "circuit_breaker_threshold": self.config.circuit_breaker_threshold,
            },
            "fallback_providers": [type(p).__name__ for p in self.fallback_providers],
            # Additional health metrics would be collected here
        }
        
    def register_default_fallback(self, interface: Type, instance: Any) -> None:
        """Register default fallback instance."""
        for provider in self.fallback_providers:
            if isinstance(provider, DefaultInstanceFallbackProvider):
                provider.register_default(interface, instance)
                return
                
        # Create new default provider if none exists
        default_provider = DefaultInstanceFallbackProvider()
        default_provider.register_default(interface, instance)
        self.add_fallback_provider(default_provider)


# Global resilience manager
_resilience_manager: Optional[ResilienceManager] = None


def get_resilience_manager() -> ResilienceManager:
    """Get global resilience manager."""
    global _resilience_manager
    if _resilience_manager is None:
        _resilience_manager = ResilienceManager()
    return _resilience_manager


def configure_resilience(config: ResilienceConfig) -> None:
    """Configure global resilience settings."""
    global _resilience_manager
    _resilience_manager = ResilienceManager(config)