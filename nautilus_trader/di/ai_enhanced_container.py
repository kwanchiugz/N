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
AI-enhanced dependency injection container.
Combines core DI functionality with intelligent service discovery and optimization.
"""

import inspect
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union
from functools import wraps
import threading
import json

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
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


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
        
        if not implementation and not factory and not instance:
            raise ValueError("Must provide implementation, factory, or instance")


class AIServiceRecommendation:
    """AI recommendation for service configuration."""
    
    def __init__(
        self,
        service_type: Type,
        recommended_lifetime: Lifetime,
        confidence: float,
        reasoning: str,
        potential_issues: List[str] = None,
    ) -> None:
        self.service_type = service_type
        self.recommended_lifetime = recommended_lifetime
        self.confidence = confidence
        self.reasoning = reasoning
        self.potential_issues = potential_issues or []
        self.timestamp = time.time()


class AIEnhancedDIContainer:
    """
    AI-enhanced dependency injection container.
    
    Provides intelligent service discovery, configuration optimization,
    and dependency analysis while maintaining simplicity.
    """
    
    def __init__(
        self,
        parent: Optional["AIEnhancedDIContainer"] = None,
        enable_ai_recommendations: bool = True,
        auto_register: bool = False,
    ) -> None:
        """
        Initialize AI-enhanced DI container.
        
        Parameters
        ----------
        parent : AIEnhancedDIContainer, optional
            Parent container for scoped containers
        enable_ai_recommendations : bool, default True
            Whether to enable AI-powered service recommendations
        auto_register : bool, default False
            Whether to enable auto-registration of services
        """
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._providers: Dict[Type, Provider] = {}
        self._parent = parent
        self._logger = Logger(self.__class__.__name__)
        self._lock = threading.RLock()
        
        # Track scoped instances
        self._scoped_instances: Dict[Type, Any] = {}
        
        # AI features
        self._enable_ai_recommendations = enable_ai_recommendations
        self._auto_register_enabled = auto_register
        self._ai_recommendations: Dict[Type, AIServiceRecommendation] = {}
        self._usage_stats: Dict[Type, Dict[str, int]] = {}
        
        # Simple performance tracking
        self._resolution_times: List[float] = []
        
    def register(
        self,
        interface: Type[T],
        implementation: Optional[Type[T]] = None,
        lifetime: Union[Lifetime, str] = Lifetime.TRANSIENT,
        factory: Optional[Callable[..., T]] = None,
        instance: Optional[T] = None,
    ) -> "AIEnhancedDIContainer":
        """Register a service in the container."""
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
                
        # Get AI recommendation if enabled
        if self._enable_ai_recommendations and implementation:
            ai_recommendation = self._get_ai_recommendation(implementation)
            if ai_recommendation and ai_recommendation.confidence > 0.8:
                if ai_recommendation.recommended_lifetime != lifetime:
                    self._logger.info(
                        f"AI recommends {ai_recommendation.recommended_lifetime.value} "
                        f"lifetime for {interface.__name__}: {ai_recommendation.reasoning}"
                    )
                
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
            
        self._logger.info(f"Registered {interface.__name__} with lifetime {lifetime.value}")
        return self
        
    def resolve(self, interface: Type[T], _resolution_chain: Optional[Set[Type]] = None) -> T:
        """Resolve a service from the container."""
        start_time = time.time()
        
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
        
        try:
            # Check this container first
            if interface in self._providers:
                provider = self._providers[interface]
                result = provider.get(self, _resolution_chain)
                self._record_usage(interface, time.time() - start_time)
                return result
                
            # Try auto-registration if enabled
            if self._auto_register_enabled and isinstance(interface, type):
                provider = self._auto_register(interface)
                result = provider.get(self, _resolution_chain)
                self._record_usage(interface, time.time() - start_time)
                return result
                
            # Check parent container
            if self._parent:
                result = self._parent.resolve(interface, _resolution_chain)
                self._record_usage(interface, time.time() - start_time)
                return result
                
            raise ResolutionError(f"Service {interface.__name__} not registered")
            
        finally:
            # Remove from resolution chain when done
            _resolution_chain.discard(interface)
    
    def _auto_register(self, cls: Type) -> Provider:
        """
        Automatically register a class with AI-enhanced defaults.
        
        Based on expert recommendations:
        - Uses FactoryProvider (transient) as safer default
        - Checks for abstract classes
        - Gets AI recommendations for optimal configuration
        """
        if inspect.isabstract(cls):
            raise TypeError(
                f"Cannot auto-register '{cls.__name__}' because it is an abstract class. "
                "Please register a concrete implementation for it explicitly."
            )
        
        # Get AI recommendation for lifetime
        recommended_lifetime = Lifetime.TRANSIENT  # Safe default
        
        if self._enable_ai_recommendations:
            ai_rec = self._get_ai_recommendation(cls)
            if ai_rec and ai_rec.confidence > 0.7:
                recommended_lifetime = ai_rec.recommended_lifetime
                self._logger.info(
                    f"AI auto-registration: {cls.__name__} as {recommended_lifetime.value} "
                    f"(confidence: {ai_rec.confidence:.2f})"
                )
        
        # Create provider based on recommendation
        if recommended_lifetime == Lifetime.SINGLETON:
            provider = SingletonProvider(ServiceDescriptor(cls, cls, lifetime=recommended_lifetime))
        elif recommended_lifetime == Lifetime.SCOPED:
            provider = ScopedProvider(ServiceDescriptor(cls, cls, lifetime=recommended_lifetime))
        else:
            provider = FactoryProvider(ServiceDescriptor(cls, cls, lifetime=recommended_lifetime))
            
        self._providers[cls] = provider
        self._services[cls] = ServiceDescriptor(cls, cls, lifetime=recommended_lifetime)
        
        self._logger.info(f"Auto-registered {cls.__name__} with lifetime {recommended_lifetime.value}")
        return provider
    
    def _get_ai_recommendation(self, service_type: Type) -> Optional[AIServiceRecommendation]:
        """Get AI recommendation for service configuration."""
        if not self._enable_ai_recommendations:
            return None
            
        # Check cache first
        if service_type in self._ai_recommendations:
            cached = self._ai_recommendations[service_type]
            # Use cached recommendation if less than 1 hour old
            if time.time() - cached.timestamp < 3600:
                return cached
        
        # Simple heuristic-based AI recommendation
        # In a real implementation, this could call actual AI services
        reasoning = []
        recommended_lifetime = Lifetime.TRANSIENT
        confidence = 0.6
        potential_issues = []
        
        # Analyze class characteristics
        class_name = service_type.__name__.lower()
        
        # Singleton patterns
        if any(pattern in class_name for pattern in ['manager', 'service', 'client', 'cache', 'pool']):
            recommended_lifetime = Lifetime.SINGLETON
            confidence = 0.8
            reasoning.append("Class name suggests singleton pattern (manager/service/client)")
            
        # Transient patterns  
        elif any(pattern in class_name for pattern in ['factory', 'builder', 'command', 'request']):
            recommended_lifetime = Lifetime.TRANSIENT
            confidence = 0.9
            reasoning.append("Class name suggests transient pattern (factory/builder/command)")
            
        # Scoped patterns
        elif any(pattern in class_name for pattern in ['context', 'session', 'transaction']):
            recommended_lifetime = Lifetime.SCOPED
            confidence = 0.8
            reasoning.append("Class name suggests scoped pattern (context/session/transaction)")
        
        # Analyze constructor complexity
        try:
            sig = inspect.signature(service_type.__init__)
            param_count = len([p for p in sig.parameters.values() if p.name != "self"])
            
            if param_count > 5:
                potential_issues.append(f"High constructor complexity ({param_count} dependencies)")
                confidence *= 0.8
                
            if param_count == 0 and confidence < 0.8:
                # Only suggest singleton for no dependencies if no strong pattern detected
                recommended_lifetime = Lifetime.SINGLETON
                reasoning.append("No dependencies suggests singleton safety")
                confidence = min(confidence + 0.1, 0.95)
                
        except Exception:
            potential_issues.append("Could not analyze constructor")
            confidence *= 0.9
        
        # Check for state indicators
        try:
            # Look for instance variables that suggest state
            if hasattr(service_type, '__annotations__'):
                annotations = service_type.__annotations__
                if any('state' in str(ann).lower() or 'cache' in str(ann).lower() 
                       for ann in annotations.values()):
                    if recommended_lifetime == Lifetime.TRANSIENT:
                        potential_issues.append("Stateful class with transient lifetime may cause issues")
                        
        except Exception:
            pass
        
        recommendation = AIServiceRecommendation(
            service_type=service_type,
            recommended_lifetime=recommended_lifetime,
            confidence=confidence,
            reasoning="; ".join(reasoning) if reasoning else "Default analysis",
            potential_issues=potential_issues,
        )
        
        # Cache the recommendation
        self._ai_recommendations[service_type] = recommendation
        return recommendation
    
    def _record_usage(self, interface: Type, resolution_time: float) -> None:
        """Record usage statistics for AI analysis."""
        if interface not in self._usage_stats:
            self._usage_stats[interface] = {"count": 0, "total_time": 0.0}
            
        self._usage_stats[interface]["count"] += 1
        self._usage_stats[interface]["total_time"] += resolution_time
        
        # Keep track of recent resolution times (last 100)
        self._resolution_times.append(resolution_time)
        if len(self._resolution_times) > 100:
            self._resolution_times.pop(0)
    
    def get_ai_insights(self) -> Dict[str, Any]:
        """Get AI insights about container usage and recommendations."""
        if not self._enable_ai_recommendations:
            return {"ai_enabled": False}
            
        insights = {
            "ai_enabled": True,
            "total_recommendations": len(self._ai_recommendations),
            "avg_resolution_time": sum(self._resolution_times) / len(self._resolution_times) if self._resolution_times else 0,
            "most_used_services": [],
            "performance_recommendations": [],
            "configuration_suggestions": [],
        }
        
        # Most used services
        sorted_usage = sorted(
            self._usage_stats.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )[:5]
        
        for service_type, stats in sorted_usage:
            avg_time = stats["total_time"] / stats["count"] if stats["count"] > 0 else 0
            insights["most_used_services"].append({
                "service": service_type.__name__,
                "usage_count": stats["count"],
                "avg_resolution_time": avg_time,
            })
            
        # Performance recommendations
        if self._resolution_times:
            avg_time = sum(self._resolution_times) / len(self._resolution_times)
            if avg_time > 0.001:  # > 1ms
                insights["performance_recommendations"].append(
                    f"Average resolution time ({avg_time*1000:.2f}ms) is high. "
                    "Consider caching or optimizing service construction."
                )
                
        # Configuration suggestions
        for service_type, recommendation in self._ai_recommendations.items():
            if recommendation.potential_issues:
                insights["configuration_suggestions"].append({
                    "service": service_type.__name__,
                    "issues": recommendation.potential_issues,
                    "recommendation": recommendation.reasoning,
                })
                
        return insights
    
    def optimize_configuration(self) -> List[str]:
        """Get AI-powered optimization suggestions."""
        suggestions = []
        
        if not self._enable_ai_recommendations:
            return ["Enable AI recommendations to get optimization suggestions"]
            
        # Analyze usage patterns
        for service_type, stats in self._usage_stats.items():
            if stats["count"] > 10:  # Frequently used
                avg_time = stats["total_time"] / stats["count"]
                if avg_time > 0.002:  # > 2ms
                    descriptor = self._services.get(service_type)
                    if descriptor and descriptor.lifetime == Lifetime.TRANSIENT:
                        suggestions.append(
                            f"Consider changing {service_type.__name__} to SINGLETON "
                            f"(used {stats['count']} times, avg {avg_time*1000:.2f}ms)"
                        )
                        
        # Check for potential issues in recommendations
        for recommendation in self._ai_recommendations.values():
            if recommendation.potential_issues:
                suggestions.extend([
                    f"{recommendation.service_type.__name__}: {issue}"
                    for issue in recommendation.potential_issues
                ])
                
        return suggestions
    
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
            
        self._providers[descriptor.interface] = provider
    
    def create_scope(self) -> "AIEnhancedDIContainer":
        """Create a scoped container."""
        return AIEnhancedDIContainer(
            parent=self,
            enable_ai_recommendations=self._enable_ai_recommendations,
            auto_register=self._auto_register_enabled,
        )
    
    def create_instance(self, cls: Type[T], _resolution_chain: Optional[Set[Type]] = None, **kwargs) -> T:
        """Create an instance with dependency injection."""
        sig = inspect.signature(cls.__init__)
        params = sig.parameters
        
        resolved_args = kwargs.copy()
        
        for name, param in params.items():
            if name == "self":
                continue
                
            if name in resolved_args:
                continue
                
            if param.annotation != param.empty:
                try:
                    resolved_args[name] = self.resolve(param.annotation, _resolution_chain)
                except ResolutionError:
                    if param.default != param.empty:
                        resolved_args[name] = param.default
                        
        return cls(**resolved_args)
    
    def print_ai_report(self) -> None:
        """Print comprehensive AI analysis report."""
        print("=" * 50)
        print("AI-Enhanced DI Container Report")
        print("=" * 50)
        
        insights = self.get_ai_insights()
        
        if not insights["ai_enabled"]:
            print("AI recommendations are disabled")
            return
            
        print(f"Total AI recommendations: {insights['total_recommendations']}")
        print(f"Average resolution time: {insights['avg_resolution_time']*1000:.2f}ms")
        print()
        
        print("Most Used Services:")
        for service in insights["most_used_services"]:
            print(f"  - {service['service']}: {service['usage_count']} uses, "
                  f"{service['avg_resolution_time']*1000:.2f}ms avg")
        print()
        
        if insights["performance_recommendations"]:
            print("Performance Recommendations:")
            for rec in insights["performance_recommendations"]:
                print(f"  - {rec}")
            print()
            
        if insights["configuration_suggestions"]:
            print("Configuration Suggestions:")
            for suggestion in insights["configuration_suggestions"]:
                print(f"  - {suggestion['service']}:")
                for issue in suggestion["issues"]:
                    print(f"    * {issue}")
                print(f"    Recommendation: {suggestion['recommendation']}")
            print()
            
        optimizations = self.optimize_configuration()
        if optimizations:
            print("Optimization Suggestions:")
            for opt in optimizations:
                print(f"  - {opt}")


# Global container instance
_container = AIEnhancedDIContainer()


def get_container() -> AIEnhancedDIContainer:
    """Get global AI-enhanced DI container."""
    return _container


def inject(func: Callable) -> Callable:
    """
    Decorator for dependency injection with AI optimization tracking.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        params = sig.parameters
        container = get_container()
        
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        
        for name, param in params.items():
            if name not in bound.arguments and param.annotation != param.empty:
                try:
                    bound.arguments[name] = container.resolve(param.annotation)
                except ResolutionError as e:
                    if param.default == param.empty:
                        raise ResolutionError(
                            f"Failed to inject required dependency '{name}' into function '{func.__name__}'",
                            interface=param.annotation,
                            original_error=e,
                        ) from e
                        
        return func(**bound.arguments)
    
    return wrapper