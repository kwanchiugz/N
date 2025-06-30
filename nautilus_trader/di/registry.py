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
Service registry for auto-discovery and registration.
"""

import importlib
import inspect
import pkgutil
from typing import Any, Dict, List, Optional, Type, Set
from pathlib import Path

from nautilus_trader.common.component import Logger
from nautilus_trader.di.container import (
    DIContainer,
    Injectable,
    Singleton,
    Transient,
    Scoped,
    Lifetime,
)
from nautilus_trader.di.exceptions import (
    ConfigurationError,
    ServiceRegistrationError,
    ModuleValidationError,
)
from nautilus_trader.di.module_validator import ModuleValidator

# Lifetime marker classes that should not be treated as service interfaces
LIFETIME_CLASSES = {Injectable, Singleton, Transient, Scoped}


class ServiceRegistry:
    """
    Registry for automatic service discovery and registration.
    
    This registry can:
    - Auto-discover services in modules
    - Register services by convention
    - Support attribute-based registration
    - Handle circular dependencies
    """
    
    def __init__(self, container: DIContainer, module_validator: Optional[ModuleValidator] = None) -> None:
        """
        Initialize service registry.
        
        Parameters
        ----------
        container : DIContainer
            Container to register services in
        module_validator : ModuleValidator, optional
            Module validator for secure imports. Uses default if not provided.
        """
        self._container = container
        self._logger = Logger(self.__class__.__name__)
        self._registered_types: Set[Type] = set()
        self._module_validator = module_validator or ModuleValidator()
        
    def scan_module(
        self,
        module_name: str,
        recursive: bool = True,
        pattern: Optional[str] = None,
    ) -> int:
        """
        Scan module for injectable services.
        
        Parameters
        ----------
        module_name : str
            Module to scan
        recursive : bool
            Whether to scan sub-modules
        pattern : str, optional
            Name pattern to match (e.g., "*Service")
            
        Returns
        -------
        int
            Number of services registered
        """
        count = 0
        
        # Validate module before importing
        try:
            self._module_validator.validate(module_name)
        except ModuleValidationError as e:
            raise ConfigurationError(
                f"Module validation failed for '{module_name}': {e}",
                module_name=module_name,
                suggestion="Ensure module is in trusted paths and contains no dangerous patterns",
            ) from e
        
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise ConfigurationError(
                f"Failed to import module '{module_name}' during auto-discovery",
                module_name=module_name,
                suggestion="Check that the module exists and has no import errors",
            ) from e
            
        # Scan module
        count += self._scan_module_members(module, pattern)
        
        # Scan sub-modules if recursive
        if recursive and hasattr(module, "__path__"):
            for _, name, _ in pkgutil.iter_modules(module.__path__):
                sub_module_name = f"{module_name}.{name}"
                count += self.scan_module(sub_module_name, recursive, pattern)
                
        return count
        
    def scan_package(
        self,
        package_path: str,
        pattern: Optional[str] = None,
    ) -> int:
        """
        Scan package directory for services.
        
        Parameters
        ----------
        package_path : str
            Path to package
        pattern : str, optional
            Name pattern to match
            
        Returns
        -------
        int
            Number of services registered
        """
        count = 0
        path = Path(package_path)
        
        if not path.exists() or not path.is_dir():
            self._logger.error(f"Invalid package path: {package_path}")
            return 0
            
        # Find all Python files
        for py_file in path.rglob("*.py"):
            if py_file.name.startswith("_"):
                continue
                
            # Convert to module name
            relative_path = py_file.relative_to(path.parent)
            module_name = str(relative_path).replace("/", ".").replace("\\", ".")[:-3]
            
            try:
                module = importlib.import_module(module_name)
                count += self._scan_module_members(module, pattern)
            except Exception as e:
                self._logger.debug(f"Failed to scan {module_name}: {e}")
                
        return count
        
    def register_assembly(self, assembly: Dict[str, Any]) -> None:
        """
        Register services from configuration assembly.
        
        Parameters
        ----------
        assembly : dict
            Service configuration
            
        Example
        -------
        assembly = {
            "services": [
                {
                    "interface": "IDataProvider",
                    "implementation": "FileDataProvider",
                    "lifetime": "singleton"
                },
                {
                    "interface": "ILogger",
                    "factory": "create_logger",
                    "lifetime": "transient"
                }
            ]
        }
        """
        services = assembly.get("services", [])
        
        for service_config in services:
            self._register_from_config(service_config)
            
    def auto_wire(self, instance: Any) -> None:
        """
        Auto-wire dependencies into an existing instance.
        
        Parameters
        ----------
        instance : Any
            Instance to wire dependencies into
        """
        # Get all attributes that are None
        for name in dir(instance):
            if name.startswith("_"):
                continue
                
            attr = getattr(instance, name)
            if attr is not None:
                continue
                
            # Try to resolve by attribute type annotation
            if hasattr(instance.__class__, "__annotations__"):
                annotations = instance.__class__.__annotations__
                if name in annotations:
                    try:
                        service = self._container.resolve(annotations[name])
                        setattr(instance, name, service)
                        self._logger.debug(
                            f"Auto-wired {name} in {instance.__class__.__name__}"
                        )
                    except ValueError:
                        pass
                        
    def _scan_module_members(
        self,
        module: Any,
        pattern: Optional[str] = None,
    ) -> int:
        """Scan module members for services."""
        count = 0
        
        for name, obj in inspect.getmembers(module):
            if not inspect.isclass(obj):
                continue
                
            # Skip if already registered
            if obj in self._registered_types:
                continue
                
            # Check if it's an injectable service
            if self._is_injectable(obj, pattern):
                if self._register_service(obj):
                    count += 1
                    self._registered_types.add(obj)
                    
        return count
        
    def _is_injectable(self, cls: Type, pattern: Optional[str] = None) -> bool:
        """Check if class should be registered."""
        # Must be a subclass of Injectable
        if not issubclass(cls, Injectable):
            return False
            
        # Check pattern if provided
        if pattern:
            import fnmatch
            if not fnmatch.fnmatch(cls.__name__, pattern):
                return False
                
        # Don't register abstract classes
        if inspect.isabstract(cls):
            return False
            
        return True
        
    def _register_service(self, cls: Type) -> bool:
        """Register a service class."""
        # Determine lifetime
        if issubclass(cls, Singleton):
            lifetime = Lifetime.SINGLETON
        elif issubclass(cls, Transient):
            lifetime = Lifetime.TRANSIENT
        elif issubclass(cls, Scoped):
            lifetime = Lifetime.SCOPED
        else:
            lifetime = Lifetime.TRANSIENT
            
        # Find interface (first base class that's Injectable but not a lifetime marker)
        interface = cls
        for base in cls.__bases__:
            if base not in LIFETIME_CLASSES and issubclass(base, Injectable):
                interface = base
                break
                
        try:
            self._container.register(
                interface=interface,
                implementation=cls,
                lifetime=lifetime,
            )
            
            self._logger.info(
                f"Registered {cls.__name__} as {interface.__name__} "
                f"with {lifetime.value} lifetime"
            )
            return True
            
        except Exception as e:
            raise ServiceRegistrationError(
                f"Failed to register service '{cls.__name__}' as '{interface.__name__}'",
                service_type=cls,
                interface_type=interface,
            ) from e
            
    def _register_from_config(self, config: Dict[str, Any]) -> None:
        """Register service from configuration."""
        interface_name = config.get("interface")
        implementation_name = config.get("implementation")
        factory_name = config.get("factory")
        lifetime = config.get("lifetime", "transient")
        
        if not interface_name:
            self._logger.error("Service config missing interface")
            return
            
        # Resolve types
        interface = self._resolve_type(interface_name)
        implementation = self._resolve_type(implementation_name) if implementation_name else None
        factory = self._resolve_callable(factory_name) if factory_name else None
        
        if not interface:
            self._logger.error(f"Could not resolve interface: {interface_name}")
            return
            
        try:
            self._container.register(
                interface=interface,
                implementation=implementation,
                lifetime=Lifetime(lifetime),
                factory=factory,
            )
        except Exception as e:
            self._logger.error(f"Failed to register {interface_name}: {e}")
            
    def _resolve_type(self, type_name: str) -> Optional[Type]:
        """Resolve type from string name."""
        parts = type_name.split(".")
        
        if len(parts) == 1:
            # Try to find in globals
            return globals().get(type_name)
            
        # Import module and get type
        module_name = ".".join(parts[:-1])
        class_name = parts[-1]
        
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except Exception:
            return None
            
    def _resolve_callable(self, callable_name: str) -> Optional[callable]:
        """Resolve callable from string name."""
        # Similar to _resolve_type but for functions
        parts = callable_name.split(".")
        
        if len(parts) == 1:
            return globals().get(callable_name)
            
        module_name = ".".join(parts[:-1])
        func_name = parts[-1]
        
        try:
            module = importlib.import_module(module_name)
            return getattr(module, func_name)
        except Exception:
            return None