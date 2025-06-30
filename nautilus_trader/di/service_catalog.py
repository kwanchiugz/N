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
Service catalog for dynamic service discovery and configuration.
"""

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, Union
from enum import Enum

from nautilus_trader.common.component import Logger
from nautilus_trader.di.container import Lifetime
from nautilus_trader.di.exceptions import ConfigurationError


class ServiceCategory(str, Enum):
    """Service category classification."""
    CORE = "core"           # Essential system services
    STORAGE = "storage"     # Data storage providers
    AUTH = "auth"          # Authentication services
    AI = "ai"              # AI/ML services
    NETWORK = "network"    # Network services
    MONITORING = "monitoring"  # Monitoring and metrics
    CUSTOM = "custom"      # User-defined services


@dataclass
class ServiceDefinition:
    """Defines a discoverable service."""
    
    name: str
    interface_path: str
    implementation_path: Optional[str] = None
    factory_path: Optional[str] = None
    lifetime: Lifetime = Lifetime.TRANSIENT
    category: ServiceCategory = ServiceCategory.CUSTOM
    dependencies: List[str] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    description: Optional[str] = None
    version: str = "1.0.0"
    
    def __post_init__(self):
        """Validate service definition."""
        if not self.implementation_path and not self.factory_path:
            raise ValueError(f"Service '{self.name}' must have either implementation_path or factory_path")


class ServiceCatalog:
    """
    Catalog of available services for dynamic discovery and configuration.
    
    This replaces hardcoded imports with a flexible, configurable system.
    """
    
    def __init__(self) -> None:
        """Initialize service catalog."""
        self._services: Dict[str, ServiceDefinition] = {}
        self._logger = Logger(self.__class__.__name__)
        self._loaded_modules: Dict[str, Any] = {}
        
        # Initialize with built-in services
        self._register_builtin_services()
        
    def register_service(self, service_def: ServiceDefinition) -> None:
        """
        Register a service definition.
        
        Parameters
        ----------
        service_def : ServiceDefinition
            Service definition to register
        """
        self._services[service_def.name] = service_def
        self._logger.debug(f"Registered service: {service_def.name}")
        
    def register_from_dict(self, config: Dict[str, Any]) -> None:
        """
        Register service from dictionary configuration.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Service configuration dictionary
        """
        service_def = ServiceDefinition(
            name=config["name"],
            interface_path=config["interface_path"],
            implementation_path=config.get("implementation_path"),
            factory_path=config.get("factory_path"),
            lifetime=Lifetime(config.get("lifetime", "transient")),
            category=ServiceCategory(config.get("category", "custom")),
            dependencies=config.get("dependencies", []),
            configuration=config.get("configuration", {}),
            enabled=config.get("enabled", True),
            description=config.get("description"),
            version=config.get("version", "1.0.0"),
        )
        self.register_service(service_def)
        
    def get_service(self, name: str) -> Optional[ServiceDefinition]:
        """Get service definition by name."""
        return self._services.get(name)
        
    def get_services_by_category(self, category: ServiceCategory) -> List[ServiceDefinition]:
        """Get all services in a category."""
        return [svc for svc in self._services.values() if svc.category == category and svc.enabled]
        
    def list_services(self, enabled_only: bool = True) -> List[ServiceDefinition]:
        """List all registered services."""
        if enabled_only:
            return [svc for svc in self._services.values() if svc.enabled]
        return list(self._services.values())
        
    def resolve_interface(self, interface_path: str) -> Type:
        """
        Resolve interface class from string path.
        
        Parameters
        ----------
        interface_path : str
            Dotted path to interface class
            
        Returns
        -------
        Type
            Interface class
            
        Raises
        ------
        ConfigurationError
            If interface cannot be resolved
        """
        try:
            module_path, class_name = interface_path.rsplit(".", 1)
            module = self._get_module(module_path)
            return getattr(module, class_name)
        except Exception as e:
            raise ConfigurationError(
                f"Failed to resolve interface '{interface_path}'",
                suggestion="Check that the interface path is correct and the module is importable",
            ) from e
            
    def resolve_implementation(self, implementation_path: str) -> Type:
        """
        Resolve implementation class from string path.
        
        Parameters
        ----------
        implementation_path : str
            Dotted path to implementation class
            
        Returns
        -------
        Type
            Implementation class
        """
        try:
            module_path, class_name = implementation_path.rsplit(".", 1)
            module = self._get_module(module_path)
            return getattr(module, class_name)
        except Exception as e:
            raise ConfigurationError(
                f"Failed to resolve implementation '{implementation_path}'",
                suggestion="Check that the implementation path is correct and the module is importable",
            ) from e
            
    def resolve_factory(self, factory_path: str) -> Callable:
        """
        Resolve factory function from string path.
        
        Parameters
        ----------
        factory_path : str
            Dotted path to factory function
            
        Returns
        -------
        Callable
            Factory function
        """
        try:
            module_path, func_name = factory_path.rsplit(".", 1)
            module = self._get_module(module_path)
            return getattr(module, func_name)
        except Exception as e:
            raise ConfigurationError(
                f"Failed to resolve factory '{factory_path}'",
                suggestion="Check that the factory path is correct and the module is importable",
            ) from e
            
    def validate_service(self, service_def: ServiceDefinition) -> bool:
        """
        Validate that a service definition is resolvable.
        
        Parameters
        ----------
        service_def : ServiceDefinition
            Service definition to validate
            
        Returns
        -------
        bool
            True if valid
            
        Raises
        ------
        ConfigurationError
            If validation fails
        """
        # Validate interface
        try:
            self.resolve_interface(service_def.interface_path)
        except ConfigurationError as e:
            raise ConfigurationError(
                f"Service '{service_def.name}' has invalid interface: {e}",
                service_class=service_def.name,
            ) from e
            
        # Validate implementation or factory
        if service_def.implementation_path:
            try:
                self.resolve_implementation(service_def.implementation_path)
            except ConfigurationError as e:
                raise ConfigurationError(
                    f"Service '{service_def.name}' has invalid implementation: {e}",
                    service_class=service_def.name,
                ) from e
                
        if service_def.factory_path:
            try:
                self.resolve_factory(service_def.factory_path)
            except ConfigurationError as e:
                raise ConfigurationError(
                    f"Service '{service_def.name}' has invalid factory: {e}",
                    service_class=service_def.name,
                ) from e
                
        return True
        
    def enable_service(self, name: str) -> None:
        """Enable a service."""
        if name in self._services:
            self._services[name].enabled = True
            self._logger.info(f"Enabled service: {name}")
            
    def disable_service(self, name: str) -> None:
        """Disable a service."""
        if name in self._services:
            self._services[name].enabled = False
            self._logger.info(f"Disabled service: {name}")
            
    def get_dependency_order(self) -> List[str]:
        """
        Get services in dependency order.
        
        Returns
        -------
        List[str]
            Service names in dependency order
            
        Raises
        ------
        ConfigurationError
            If circular dependencies detected
        """
        # Simple topological sort
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(service_name: str):
            if service_name in temp_visited:
                raise ConfigurationError(
                    f"Circular dependency detected involving service '{service_name}'",
                    suggestion="Review service dependencies to eliminate cycles",
                )
            if service_name in visited:
                return
                
            temp_visited.add(service_name)
            
            service = self._services.get(service_name)
            if service and service.enabled:
                for dep in service.dependencies:
                    if dep in self._services:
                        visit(dep)
                        
            temp_visited.remove(service_name)
            visited.add(service_name)
            order.append(service_name)
            
        for service_name in self._services:
            if service_name not in visited and self._services[service_name].enabled:
                visit(service_name)
                
        return order
        
    def _get_module(self, module_path: str) -> Any:
        """Get module, using cache for efficiency."""
        if module_path not in self._loaded_modules:
            self._loaded_modules[module_path] = importlib.import_module(module_path)
        return self._loaded_modules[module_path]
        
    def _register_builtin_services(self) -> None:
        """Register built-in Nautilus services."""
        
        # Core services
        self.register_service(ServiceDefinition(
            name="event_bus",
            interface_path="nautilus_trader.common.event_bus.EventBus",
            factory_path="nautilus_trader.common.event_bus.get_event_bus",
            lifetime=Lifetime.SINGLETON,
            category=ServiceCategory.CORE,
            description="Central event bus for system-wide messaging",
        ))
        
        # Storage services
        self.register_service(ServiceDefinition(
            name="file_storage",
            interface_path="nautilus_trader.storage.StorageProvider",
            implementation_path="nautilus_trader.storage.FileStorageProvider",
            lifetime=Lifetime.TRANSIENT,
            category=ServiceCategory.STORAGE,
            description="File-based storage provider",
        ))
        
        self.register_service(ServiceDefinition(
            name="database_storage",
            interface_path="nautilus_trader.storage.StorageProvider",
            implementation_path="nautilus_trader.storage.DatabaseStorageProvider",
            lifetime=Lifetime.TRANSIENT,
            category=ServiceCategory.STORAGE,
            description="Database storage provider",
        ))
        
        self.register_service(ServiceDefinition(
            name="redis_storage",
            interface_path="nautilus_trader.storage.StorageProvider",
            implementation_path="nautilus_trader.storage.RedisStorageProvider",
            lifetime=Lifetime.TRANSIENT,
            category=ServiceCategory.STORAGE,
            description="Redis storage provider",
        ))
        
        self.register_service(ServiceDefinition(
            name="s3_storage",
            interface_path="nautilus_trader.storage.StorageProvider",
            implementation_path="nautilus_trader.storage.S3StorageProvider",
            lifetime=Lifetime.TRANSIENT,
            category=ServiceCategory.STORAGE,
            description="AWS S3 storage provider",
        ))
        
        # Default storage with factory
        self.register_service(ServiceDefinition(
            name="default_storage",
            interface_path="nautilus_trader.storage.StorageProvider",
            factory_path="nautilus_trader.storage.StorageFactory.create_default",
            lifetime=Lifetime.SINGLETON,
            category=ServiceCategory.STORAGE,
            description="Default storage provider (file-based)",
        ))
        
        # Auth services
        self.register_service(ServiceDefinition(
            name="session_manager",
            interface_path="nautilus_trader.auth.session_v2.SessionManagerV2",
            implementation_path="nautilus_trader.auth.session_v2.SessionManagerV2",
            lifetime=Lifetime.SINGLETON,
            category=ServiceCategory.AUTH,
            dependencies=["default_storage"],
            description="JWT-based session management",
        ))
        
        # AI services
        self.register_service(ServiceDefinition(
            name="deepseek_provider",
            interface_path="nautilus_trader.ai.providers.AIProvider",
            implementation_path="nautilus_trader.ai.providers.DeepSeekProvider",
            lifetime=Lifetime.SINGLETON,
            category=ServiceCategory.AI,
            description="DeepSeek AI provider",
        ))


# Global service catalog instance
_service_catalog = ServiceCatalog()


def get_service_catalog() -> ServiceCatalog:
    """Get the global service catalog instance."""
    return _service_catalog


def register_service(service_def: ServiceDefinition) -> None:
    """Register a service in the global catalog."""
    _service_catalog.register_service(service_def)


def register_service_from_config(config: Dict[str, Any]) -> None:
    """Register a service from configuration dictionary."""
    _service_catalog.register_from_dict(config)