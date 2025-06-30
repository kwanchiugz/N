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
Dependency injection container implementation.
"""

import inspect
import time
from abc import ABC
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union
from functools import wraps
import threading
import asyncio

from nautilus_trader.common.component import Logger
from nautilus_trader.di.providers import (
    Provider,
    SingletonProvider,
    TransientProvider,
    ScopedProvider,
    FactoryProvider,
)
from nautilus_trader.di.exceptions import ResolutionError, CircularDependencyError


T = TypeVar("T")


class Lifetime(str, Enum):
    """Service lifetime options."""
    SINGLETON = "singleton"    # Single instance for application lifetime
    TRANSIENT = "transient"    # New instance every time
    SCOPED = "scoped"         # Single instance per scope


class Injectable(ABC):
    """Base class for injectable services."""
    pass


class Singleton(Injectable):
    """Marker for singleton services."""
    pass


class Transient(Injectable):
    """Marker for transient services."""
    pass


class Scoped(Injectable):
    """Marker for scoped services."""
    pass


class ServiceDescriptor:
    """Describes a registered service."""
    
    def __init__(
        self,
        interface: Type,
        implementation: Optional[Type] = None,
        factory: Optional[Callable] = None,
        lifetime: Lifetime = Lifetime.TRANSIENT,
        instance: Optional[Any] = None,
    ) -> None:
        self.interface = interface
        self.implementation = implementation or interface
        self.factory = factory
        self.lifetime = lifetime
        self.instance = instance
        
        # Validate
        if not implementation and not factory and not instance:
            raise ValueError("Must provide implementation, factory, or instance")


class DIContainer:
    """
    Dependency injection container.
    
    This container provides:
    - Service registration with lifetimes
    - Automatic dependency resolution
    - Constructor injection
    - Factory support
    - Scoped containers
    """
    
    def __init__(
        self,
        parent: Optional["DIContainer"] = None,
        config: Optional[Any] = None,
        enable_monitoring: Optional[bool] = None,
    ) -> None:
        """
        Initialize DI container.
        
        Parameters
        ----------
        parent : DIContainer, optional
            Parent container for scoped containers
        config : DIContainerConfig, optional
            Container configuration
        enable_monitoring : bool, optional
            Whether to enable performance monitoring (overrides config if provided)
        """
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._providers: Dict[Type, Provider] = {}
        self._parent = parent
        self._logger = Logger(self.__class__.__name__)
        self._lock = threading.RLock()
        
        # Track scoped instances
        self._scoped_instances: Dict[Type, Any] = {}
        
        # Configuration
        self._config = config
        if not config and not parent:
            # Load default configuration
            try:
                from nautilus_trader.di.config import get_config
                self._config = get_config()
            except ImportError:
                self._logger.warning("Configuration system not available, using defaults")
        
        # Feature flags from config
        monitoring_enabled = enable_monitoring
        if monitoring_enabled is None and self._config:
            monitoring_enabled = self._config.monitoring.enabled
        elif monitoring_enabled is None:
            monitoring_enabled = True
            
        # Monitoring integration
        self._monitoring_enabled = monitoring_enabled
        self._metrics_collector = None
        if monitoring_enabled and not parent:  # Only root container collects metrics
            self._initialize_monitoring()
            
        # Caching integration
        self._caching_enabled = self._config.caching.enabled if self._config else True
        self._service_cache = None
        self._cache_manager = None
        if self._caching_enabled and not parent:  # Only root container manages cache
            self._initialize_caching()
            
        # Resilience integration
        self._resilience_enabled = self._config.resilience.enabled if self._config else True
        self._resilience_manager = None
        if self._resilience_enabled and not parent:  # Only root container manages resilience
            self._initialize_resilience()
        
    def register(
        self,
        interface: Type[T],
        implementation: Optional[Type[T]] = None,
        lifetime: Union[Lifetime, str] = Lifetime.TRANSIENT,
        factory: Optional[Callable[..., T]] = None,
        instance: Optional[T] = None,
    ) -> "DIContainer":
        """
        Register a service in the container.
        
        Parameters
        ----------
        interface : Type
            Service interface or base class
        implementation : Type, optional
            Concrete implementation
        lifetime : Lifetime or str
            Service lifetime
        factory : Callable, optional
            Factory function to create instances
        instance : Any, optional
            Pre-created instance (for singleton)
            
        Returns
        -------
        DIContainer
            Self for chaining
        """
        if isinstance(lifetime, str):
            lifetime = Lifetime(lifetime)
            
        # Auto-detect lifetime from base class
        if implementation and not factory and not instance:
            if issubclass(implementation, Singleton):
                lifetime = Lifetime.SINGLETON
            elif issubclass(implementation, Transient):
                lifetime = Lifetime.TRANSIENT
            elif issubclass(implementation, Scoped):
                lifetime = Lifetime.SCOPED
                
        descriptor = ServiceDescriptor(
            interface=interface,
            implementation=implementation,
            factory=factory,
            lifetime=lifetime,
            instance=instance,
        )
        
        with self._lock:
            self._services[interface] = descriptor
            self._create_provider(descriptor)
            
        self._logger.info(
            f"Registered {interface.__name__} with lifetime {lifetime.value}"
        )
        
        return self
        
    def register_singleton(
        self,
        interface: Type[T],
        implementation: Optional[Type[T]] = None,
        factory: Optional[Callable[..., T]] = None,
        instance: Optional[T] = None,
    ) -> "DIContainer":
        """Register a singleton service."""
        return self.register(
            interface,
            implementation,
            Lifetime.SINGLETON,
            factory,
            instance,
        )
        
    def register_transient(
        self,
        interface: Type[T],
        implementation: Optional[Type[T]] = None,
        factory: Optional[Callable[..., T]] = None,
    ) -> "DIContainer":
        """Register a transient service."""
        return self.register(
            interface,
            implementation,
            Lifetime.TRANSIENT,
            factory,
        )
        
    def register_scoped(
        self,
        interface: Type[T],
        implementation: Optional[Type[T]] = None,
        factory: Optional[Callable[..., T]] = None,
    ) -> "DIContainer":
        """Register a scoped service."""
        return self.register(
            interface,
            implementation,
            Lifetime.SCOPED,
            factory,
        )
        
    def resolve(self, interface: Type[T], _resolution_chain: Optional[Set[Type]] = None) -> T:
        """
        Resolve a service from the container.
        
        Parameters
        ----------
        interface : Type
            Service interface to resolve
        _resolution_chain : Set[Type], optional
            Internal parameter to track resolution chain for circular dependency detection
            
        Returns
        -------
        T
            Service instance
            
        Raises
        ------
        ValueError
            If service not registered
        CircularDependencyError
            If circular dependency detected
        """
        # Initialize resolution chain for root call
        if _resolution_chain is None:
            _resolution_chain = set()
            
        # Check for circular dependency
        if interface in _resolution_chain:
            cycle_path = list(_resolution_chain) + [interface]
            cycle_str = " -> ".join(t.__name__ for t in cycle_path)
            raise CircularDependencyError(
                f"Circular dependency detected: {cycle_str}",
                cycle_path=cycle_path,
            )
            
        # Add current interface to resolution chain
        _resolution_chain.add(interface)
        
        # Performance monitoring
        start_time = time.time() if self._monitoring_enabled else None
        success = False
        
        try:
            # Check this container first
            if interface in self._providers:
                provider = self._providers[interface]
                result = provider.get(self, _resolution_chain)
                success = True
                return result
                
            # Check parent container
            if self._parent:
                result = self._parent.resolve(interface, _resolution_chain)
                success = True
                return result
                
            raise ValueError(f"Service {interface.__name__} not registered")
            
        finally:
            # Remove from resolution chain when done
            _resolution_chain.discard(interface)
            
            # Record metrics if monitoring enabled
            if self._monitoring_enabled and start_time is not None:
                duration = time.time() - start_time
                self._record_resolution_metrics(interface.__name__, duration, success)
        
    def resolve_all(self, interface: Type[T]) -> List[T]:
        """
        Resolve all services implementing an interface.
        
        Parameters
        ----------
        interface : Type
            Service interface
            
        Returns
        -------
        List[T]
            All service instances
        """
        instances = []
        
        # Collect from this container
        for service_type, provider in self._providers.items():
            if issubclass(service_type, interface):
                instances.append(provider.get(self))
                
        # Collect from parent
        if self._parent:
            instances.extend(self._parent.resolve_all(interface))
            
        return instances
        
    def create_scope(self) -> "DIContainer":
        """
        Create a scoped container.
        
        Returns
        -------
        DIContainer
            Scoped container
        """
        return DIContainer(parent=self)
        
    def create_instance(self, cls: Type[T], _resolution_chain: Optional[Set[Type]] = None, **kwargs) -> T:
        """
        Create an instance with dependency injection.
        
        Parameters
        ----------
        cls : Type
            Class to instantiate
        _resolution_chain : Set[Type], optional
            Internal parameter for circular dependency detection
        **kwargs
            Additional constructor arguments
            
        Returns
        -------
        T
            Created instance
        """
        # Get constructor signature
        sig = inspect.signature(cls.__init__)
        params = sig.parameters
        
        # Resolve dependencies
        resolved_args = kwargs.copy()
        
        for name, param in params.items():
            if name == "self":
                continue
                
            # Skip if already provided
            if name in resolved_args:
                continue
                
            # Try to resolve by type annotation
            if param.annotation != param.empty:
                try:
                    resolved_args[name] = self.resolve(param.annotation, _resolution_chain)
                except ValueError:
                    # Use default if available
                    if param.default != param.empty:
                        resolved_args[name] = param.default
                    # Otherwise let it fail naturally
                    
        return cls(**resolved_args)
        
    def _create_provider(self, descriptor: ServiceDescriptor) -> None:
        """Create appropriate provider for service."""
        if descriptor.lifetime == Lifetime.SINGLETON:
            provider = SingletonProvider(descriptor)
        elif descriptor.lifetime == Lifetime.TRANSIENT:
            provider = TransientProvider(descriptor)
        elif descriptor.lifetime == Lifetime.SCOPED:
            provider = ScopedProvider(descriptor)
        else:
            raise ValueError(f"Unknown lifetime: {descriptor.lifetime}")
            
        # Wrap with caching if enabled and appropriate
        if self._should_cache_service(descriptor):
            provider = self._wrap_with_cache(provider)
            
        # Wrap with resilience if enabled
        if self._should_add_resilience(descriptor):
            provider = self._wrap_with_resilience(provider)
            
        self._providers[descriptor.interface] = provider
        
    def _initialize_monitoring(self) -> None:
        """Initialize monitoring for this container."""
        try:
            from nautilus_trader.di.monitoring import initialize_monitoring, get_metrics_collector
            initialize_monitoring(self)
            self._metrics_collector = get_metrics_collector()
            self._logger.info("Monitoring initialized")
        except ImportError:
            self._logger.warning("Monitoring dependencies not available, disabling monitoring")
            self._monitoring_enabled = False
            
    def _record_resolution_metrics(self, service_name: str, duration: float, success: bool) -> None:
        """Record resolution metrics."""
        if self._metrics_collector:
            self._metrics_collector.record_resolution(service_name, duration, success)
        elif self._parent and hasattr(self._parent, '_record_resolution_metrics'):
            # Delegate to parent container
            self._parent._record_resolution_metrics(service_name, duration, success)
            
    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """
        Get monitoring dashboard data.
        
        Returns
        -------
        Dict[str, Any]
            Dashboard data if monitoring is enabled
            
        Raises
        ------
        RuntimeError
            If monitoring is not enabled
        """
        if not self._monitoring_enabled:
            raise RuntimeError("Monitoring is not enabled for this container")
            
        try:
            from nautilus_trader.di.monitoring import get_dashboard
            dashboard = get_dashboard()
            if dashboard:
                return dashboard.get_dashboard_data()
            else:
                raise RuntimeError("Monitoring dashboard not initialized")
        except ImportError:
            raise RuntimeError("Monitoring dependencies not available")
            
    def get_health_status(self) -> str:
        """
        Get overall health status.
        
        Returns
        -------
        str
            Health status: 'healthy', 'degraded', 'unhealthy', or 'unknown'
        """
        if not self._monitoring_enabled:
            return "unknown"
            
        try:
            from nautilus_trader.di.monitoring import get_health_checker
            health_checker = get_health_checker()
            if health_checker:
                return health_checker.get_overall_health().value
            else:
                return "unknown"
        except ImportError:
            return "unknown"
            
    def print_monitoring_dashboard(self) -> None:
        """Print monitoring dashboard to console."""
        if not self._monitoring_enabled:
            print("Monitoring is not enabled for this container")
            return
            
        try:
            from nautilus_trader.di.monitoring import get_dashboard
            dashboard = get_dashboard()
            if dashboard:
                print(dashboard.format_dashboard_text())
            else:
                print("Monitoring dashboard not initialized")
        except ImportError:
            print("Monitoring dependencies not available")
            
    def _initialize_caching(self) -> None:
        """Initialize caching for this container."""
        try:
            from nautilus_trader.di.caching import get_cache_manager, create_default_cache
            self._cache_manager = get_cache_manager()
            self._service_cache = create_default_cache()
            self._cache_manager.start_cleanup_thread()
            self._logger.info("Caching initialized")
        except ImportError:
            self._logger.warning("Caching dependencies not available, disabling caching")
            self._caching_enabled = False
            
    def _should_cache_service(self, descriptor: ServiceDescriptor) -> bool:
        """Determine if a service should be cached."""
        if not self._caching_enabled or not self._service_cache:
            return False
            
        # Cache singletons and scoped services, but not transients
        return descriptor.lifetime in [Lifetime.SINGLETON, Lifetime.SCOPED]
        
    def _wrap_with_cache(self, provider: Provider) -> Provider:
        """Wrap provider with caching."""
        try:
            from nautilus_trader.di.caching import CachedServiceProvider
            return CachedServiceProvider(provider, self._service_cache)
        except ImportError:
            self._logger.warning("Cannot wrap provider with cache - caching disabled")
            return provider
            
    def clear_service_cache(self) -> None:
        """Clear service resolution cache."""
        if self._service_cache:
            self._service_cache.clear()
            self._logger.info("Service cache cleared")
        elif self._parent and hasattr(self._parent, 'clear_service_cache'):
            self._parent.clear_service_cache()
            
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns
        -------
        Dict[str, Any]
            Cache statistics if caching is enabled
        """
        if self._service_cache:
            return self._service_cache.get_stats()
        elif self._parent and hasattr(self._parent, 'get_cache_stats'):
            return self._parent.get_cache_stats()
        else:
            return {"caching_enabled": False}
            
    def configure_cache(
        self,
        max_size: Optional[int] = None,
        cleanup_interval: Optional[float] = None,
    ) -> None:
        """
        Configure cache settings.
        
        Parameters
        ----------
        max_size : int, optional
            Maximum cache size
        cleanup_interval : float, optional
            Cleanup interval in seconds
        """
        if not self._caching_enabled:
            self._logger.warning("Caching is not enabled")
            return
            
        if max_size and self._service_cache:
            self._service_cache.resize(max_size)
            
        if cleanup_interval and self._cache_manager:
            self._cache_manager._cleanup_interval = cleanup_interval
            
        self._logger.info(f"Cache configured: max_size={max_size}, cleanup_interval={cleanup_interval}")
        
    def print_cache_stats(self) -> None:
        """Print cache statistics to console."""
        stats = self.get_cache_stats()
        
        if not stats.get("caching_enabled", True):
            print("Caching is not enabled")
            return
            
        print("Service Cache Statistics:")
        print("=" * 30)
        print(f"Hits: {stats.get('hits', 0)}")
        print(f"Misses: {stats.get('misses', 0)}")
        print(f"Hit Ratio: {stats.get('hit_ratio', 0.0):.1%}")
        print(f"Size: {stats.get('size', 0)}/{stats.get('max_size', 0)}")
        print(f"Evictions: {stats.get('evictions', 0)}")
        print(f"Strategy: {stats.get('strategy', 'unknown')}")
        
    def _initialize_resilience(self) -> None:
        """Initialize resilience for this container."""
        try:
            from nautilus_trader.di.resilience import get_resilience_manager
            self._resilience_manager = get_resilience_manager()
            self._logger.info("Resilience initialized")
        except ImportError:
            self._logger.warning("Resilience dependencies not available, disabling resilience")
            self._resilience_enabled = False
            
    def _should_add_resilience(self, descriptor: ServiceDescriptor) -> bool:
        """Determine if a service should have resilience features."""
        if not self._resilience_enabled or not self._resilience_manager:
            return False
            
        # Add resilience for external dependencies or critical services
        # For now, add resilience to all services except simple value types
        return not self._is_simple_value_type(descriptor.interface)
        
    def _is_simple_value_type(self, interface: Type) -> bool:
        """Check if interface represents a simple value type."""
        # Simple heuristic - could be made more sophisticated
        simple_types = {str, int, float, bool, list, dict, set, tuple}
        return interface in simple_types or interface.__name__.endswith('Config')
        
    def _wrap_with_resilience(self, provider: Provider) -> Provider:
        """Wrap provider with resilience."""
        try:
            return self._resilience_manager.wrap_provider(provider)
        except Exception as e:
            self._logger.warning(f"Cannot wrap provider with resilience: {e}")
            return provider
            
    def configure_resilience(
        self,
        failure_mode: Optional[str] = None,
        max_retries: Optional[int] = None,
        circuit_breaker_threshold: Optional[int] = None,
    ) -> None:
        """
        Configure resilience settings.
        
        Parameters
        ----------
        failure_mode : str, optional
            Failure mode: 'fail_fast', 'graceful', 'circuit_breaker', 'retry_backoff'
        max_retries : int, optional
            Maximum retry attempts
        circuit_breaker_threshold : int, optional
            Circuit breaker failure threshold
        """
        if not self._resilience_enabled:
            self._logger.warning("Resilience is not enabled")
            return
            
        if self._resilience_manager:
            config = self._resilience_manager.config
            
            if failure_mode:
                from nautilus_trader.di.resilience import FailureMode
                config.failure_mode = FailureMode(failure_mode)
                
            if max_retries is not None:
                config.max_retries = max_retries
                
            if circuit_breaker_threshold is not None:
                config.circuit_breaker_threshold = circuit_breaker_threshold
                
            self._logger.info(f"Resilience configured: mode={failure_mode}, retries={max_retries}")
            
    def register_fallback(self, interface: Type[T], fallback_instance: T) -> None:
        """
        Register fallback instance for service.
        
        Parameters
        ----------
        interface : Type[T]
            Service interface
        fallback_instance : T
            Fallback instance to use if primary service fails
        """
        if not self._resilience_enabled or not self._resilience_manager:
            self._logger.warning("Resilience is not enabled, cannot register fallback")
            return
            
        self._resilience_manager.register_default_fallback(interface, fallback_instance)
        self._logger.info(f"Registered fallback for {interface.__name__}")
        
    def get_service_health_report(self) -> Dict[str, Any]:
        """
        Get health report for all services.
        
        Returns
        -------
        Dict[str, Any]
            Health report if resilience is enabled
        """
        if not self._resilience_enabled or not self._resilience_manager:
            return {"resilience_enabled": False}
            
        return self._resilience_manager.get_health_report()
        
    def print_service_health(self) -> None:
        """Print service health report to console."""
        report = self.get_service_health_report()
        
        if not report.get("resilience_enabled", True):
            print("Resilience is not enabled")
            return
            
        print("Service Health Report:")
        print("=" * 30)
        
        config = report.get("config", {})
        print(f"Failure Mode: {config.get('failure_mode', 'unknown')}")
        print(f"Max Retries: {config.get('max_retries', 'unknown')}")
        print(f"Circuit Breaker Threshold: {config.get('circuit_breaker_threshold', 'unknown')}")
        
        fallback_providers = report.get("fallback_providers", [])
        if fallback_providers:
            print(f"Fallback Providers: {', '.join(fallback_providers)}")
            
    def get_configuration(self) -> Optional[Any]:
        """Get current container configuration."""
        return self._config
        
    def update_configuration(self, config: Any) -> None:
        """
        Update container configuration.
        
        Parameters
        ----------
        config : DIContainerConfig
            New configuration
        """
        self._config = config
        
        # Apply configuration changes
        if config.monitoring.enabled != self._monitoring_enabled:
            self._monitoring_enabled = config.monitoring.enabled
            if self._monitoring_enabled and not self._metrics_collector:
                self._initialize_monitoring()
                
        if config.caching.enabled != self._caching_enabled:
            self._caching_enabled = config.caching.enabled
            if self._caching_enabled and not self._service_cache:
                self._initialize_caching()
                
        if config.resilience.enabled != self._resilience_enabled:
            self._resilience_enabled = config.resilience.enabled
            if self._resilience_enabled and not self._resilience_manager:
                self._initialize_resilience()
                
        self._logger.info("Container configuration updated")
        
    def apply_config_overrides(self, **overrides) -> None:
        """
        Apply configuration overrides.
        
        Parameters
        ----------
        **overrides
            Configuration field overrides
        """
        if not self._config:
            self._logger.warning("No configuration available for overrides")
            return
            
        # Apply overrides using dot notation
        for key, value in overrides.items():
            try:
                # Parse dot notation (e.g., "monitoring.enabled" -> config.monitoring.enabled)
                parts = key.split('.')
                obj = self._config
                
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                    
                setattr(obj, parts[-1], value)
                self._logger.debug(f"Applied config override: {key} = {value}")
                
            except AttributeError:
                self._logger.warning(f"Invalid config path: {key}")
                
        # Re-apply configuration
        self.update_configuration(self._config)
        
    def print_configuration(self) -> None:
        """Print current configuration to console."""
        if not self._config:
            print("No configuration available")
            return
            
        print("DI Container Configuration:")
        print("=" * 40)
        print(f"Name: {self._config.container_name}")
        print(f"Version: {self._config.version}")
        
        if self._config.description:
            print(f"Description: {self._config.description}")
            
        print("\nComponent Status:")
        print(f"  Monitoring: {'Enabled' if self._monitoring_enabled else 'Disabled'}")
        print(f"  Caching: {'Enabled' if self._caching_enabled else 'Disabled'}")
        print(f"  Resilience: {'Enabled' if self._resilience_enabled else 'Disabled'}")
        
        print(f"\nSecurity Settings:")
        print(f"  Strict Mode: {self._config.security.strict_mode}")
        print(f"  Audit Imports: {self._config.security.audit_imports}")
        print(f"  Trusted Prefixes: {len(self._config.security.trusted_prefixes)}")
        
        print(f"\nValidation Settings:")
        print(f"  Graph Validation: {self._config.validation.enable_graph_validation}")
        print(f"  Max Constructor Dependencies: {self._config.validation.max_constructor_dependencies}")
        print(f"  Circular Detection: {self._config.validation.enable_circular_detection}")
        
        if self._caching_enabled:
            print(f"\nCache Settings:")
            print(f"  Strategy: {self._config.caching.default_strategy.value}")
            print(f"  Max Size: {self._config.caching.default_max_size}")
            print(f"  TTL: {self._config.caching.default_ttl_seconds}s")
            
        if self._resilience_enabled:
            print(f"\nResilience Settings:")
            print(f"  Failure Mode: {self._config.resilience.default_failure_mode.value}")
            print(f"  Max Retries: {self._config.resilience.max_retries}")
            print(f"  Circuit Breaker Threshold: {self._config.resilience.circuit_breaker_threshold}")
            
    def _resolve_dependencies(self, cls: Type) -> Dict[str, Any]:
        """Resolve constructor dependencies for a class."""
        sig = inspect.signature(cls.__init__)
        params = sig.parameters
        
        dependencies = {}
        for name, param in params.items():
            if name == "self":
                continue
                
            # Try to resolve by type annotation
            if param.annotation != param.empty:
                try:
                    dependencies[name] = self.resolve(param.annotation)
                except ValueError:
                    # Use default if available
                    if param.default != param.empty:
                        dependencies[name] = param.default
                        
        return dependencies


# Global container instance
_container = DIContainer()


def get_container() -> DIContainer:
    """Get global DI container."""
    return _container


def inject(func: Callable) -> Callable:
    """
    Decorator for dependency injection.
    
    Automatically injects dependencies based on type annotations.
    
    Example
    -------
    @inject
    def process_data(analyzer: DataAnalyzer, storage: StorageProvider):
        # analyzer and storage are automatically injected
        pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get function signature
        sig = inspect.signature(func)
        params = sig.parameters
        
        # Get container
        container = get_container()
        
        # Resolve missing parameters
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        
        for name, param in params.items():
            if name not in bound.arguments and param.annotation != param.empty:
                try:
                    bound.arguments[name] = container.resolve(param.annotation)
                except ValueError as e:
                    # If parameter has no default, this is a critical error
                    if param.default == param.empty:
                        raise ResolutionError(
                            f"Failed to inject required dependency '{name}' into function '{func.__name__}'",
                            interface=param.annotation,
                            original_error=e,
                        ) from e
                    # Otherwise let the function use its default value
                    
        return func(**bound.arguments)
        
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        # Same logic for async functions
        sig = inspect.signature(func)
        params = sig.parameters
        container = get_container()
        
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        
        for name, param in params.items():
            if name not in bound.arguments and param.annotation != param.empty:
                try:
                    bound.arguments[name] = container.resolve(param.annotation)
                except ValueError as e:
                    # If parameter has no default, this is a critical error
                    if param.default == param.empty:
                        raise ResolutionError(
                            f"Failed to inject required dependency '{name}' into async function '{func.__name__}'",
                            interface=param.annotation,
                            original_error=e,
                        ) from e
                    # Otherwise let the function use its default value
                    
        return await func(**bound.arguments)
        
    # Return appropriate wrapper
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return wrapper


# Configuration helpers
class ContainerBuilder:
    """
    Builder for configuring DI container.
    
    Provides fluent API for service registration.
    """
    
    def __init__(self, container: Optional[DIContainer] = None) -> None:
        self._container = container or DIContainer()
        
    def add_singleton(
        self,
        interface: Type[T],
        implementation: Optional[Type[T]] = None,
        factory: Optional[Callable[..., T]] = None,
        instance: Optional[T] = None,
    ) -> "ContainerBuilder":
        """Add singleton service."""
        self._container.register_singleton(interface, implementation, factory, instance)
        return self
        
    def add_transient(
        self,
        interface: Type[T],
        implementation: Optional[Type[T]] = None,
        factory: Optional[Callable[..., T]] = None,
    ) -> "ContainerBuilder":
        """Add transient service."""
        self._container.register_transient(interface, implementation, factory)
        return self
        
    def add_scoped(
        self,
        interface: Type[T],
        implementation: Optional[Type[T]] = None,
        factory: Optional[Callable[..., T]] = None,
    ) -> "ContainerBuilder":
        """Add scoped service."""
        self._container.register_scoped(interface, implementation, factory)
        return self
        
    def build(self) -> DIContainer:
        """Build and return container."""
        return self._container