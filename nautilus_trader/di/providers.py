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
Service providers for dependency injection.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Set, Type, TYPE_CHECKING
import threading

if TYPE_CHECKING:
    from nautilus_trader.di.container import ServiceDescriptor, DIContainer


class Provider(ABC):
    """Abstract base class for service providers."""
    
    def __init__(self, descriptor: "ServiceDescriptor") -> None:
        self._descriptor = descriptor
        
    @abstractmethod
    def get(self, container: "DIContainer", resolution_chain: Optional[Set[Type]] = None) -> Any:
        """Get service instance."""
        pass


class SingletonProvider(Provider):
    """
    Provider for singleton services.
    
    Creates and caches a single instance.
    """
    
    def __init__(self, descriptor: "ServiceDescriptor") -> None:
        super().__init__(descriptor)
        self._instance = descriptor.instance
        self._lock = threading.Lock()
        
    def get(self, container: "DIContainer", resolution_chain: Optional[Set[Type]] = None) -> Any:
        """Get singleton instance."""
        if self._instance is None:
            with self._lock:
                # Double-check pattern
                if self._instance is None:
                    self._instance = self._create_instance(container, resolution_chain)
        return self._instance
        
    def _create_instance(self, container: "DIContainer", resolution_chain: Optional[Set[Type]] = None) -> Any:
        """Create the singleton instance."""
        if self._descriptor.factory:
            # Use factory
            return self._descriptor.factory()
        else:
            # Create with dependency injection
            return container.create_instance(self._descriptor.implementation, resolution_chain)


class TransientProvider(Provider):
    """
    Provider for transient services.
    
    Creates a new instance every time.
    """
    
    def get(self, container: "DIContainer", resolution_chain: Optional[Set[Type]] = None) -> Any:
        """Get new instance."""
        if self._descriptor.factory:
            # Use factory
            return self._descriptor.factory()
        else:
            # Create with dependency injection
            return container.create_instance(self._descriptor.implementation, resolution_chain)


class ScopedProvider(Provider):
    """
    Provider for scoped services.
    
    Creates one instance per scope/container.
    """
    
    def get(self, container: "DIContainer", resolution_chain: Optional[Set[Type]] = None) -> Any:
        """Get scoped instance."""
        # Check if instance exists in container's scope
        if self._descriptor.interface in container._scoped_instances:
            return container._scoped_instances[self._descriptor.interface]
            
        # Create new instance for this scope
        if self._descriptor.factory:
            instance = self._descriptor.factory()
        else:
            instance = container.create_instance(self._descriptor.implementation, resolution_chain)
            
        # Cache in container's scope
        container._scoped_instances[self._descriptor.interface] = instance
        return instance


class FactoryProvider(Provider):
    """
    Provider that always uses a factory function.
    
    Can be used for complex creation logic.
    """
    
    def get(self, container: "DIContainer", resolution_chain: Optional[Set[Type]] = None) -> Any:
        """Get instance from factory."""
        return self._descriptor.factory(container)