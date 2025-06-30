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
Tests for the registry interface detection fix.
"""

import pytest

from nautilus_trader.di.container import DIContainer, Injectable, Singleton, Transient, Scoped
from nautilus_trader.di.registry import ServiceRegistry


# Test interfaces
class IDataService(Injectable):
    """Test data service interface."""
    pass


class ILogService(Injectable):
    """Test log service interface."""
    pass


class ICacheService(Injectable):
    """Test cache service interface."""
    pass


# Test implementations with different inheritance orders
class DataServiceTransientFirst(Transient, IDataService):
    """Service with lifetime marker first."""
    pass


class DataServiceInterfaceFirst(IDataService, Transient):
    """Service with interface first."""
    pass


class LogServiceSingleton(Singleton, ILogService):
    """Singleton service with lifetime marker first."""
    pass


class CacheServiceScoped(ICacheService, Scoped):
    """Scoped service with interface first."""
    pass


class StandaloneTransient(Transient):
    """Service with only lifetime marker, no explicit interface."""
    pass


class TestRegistryInterfaceFix:
    """Test the fixed interface detection in ServiceRegistry."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.container = DIContainer()
        self.registry = ServiceRegistry(self.container)
        
    def test_transient_first_registers_correctly(self):
        """Test that service with Transient first registers to correct interface."""
        # Act
        count = self.registry._scan_module_members(
            module=type(self).__module__, 
            pattern="DataServiceTransientFirst"
        )
        
        # Assert
        assert count == 1
        # Should be able to resolve by IDataService, not Transient
        service = self.container.resolve(IDataService)
        assert isinstance(service, DataServiceTransientFirst)
        
        # Should NOT be registered as Transient
        with pytest.raises(ValueError, match="Service.*Transient.*not registered"):
            self.container.resolve(Transient)
            
    def test_interface_first_registers_correctly(self):
        """Test that service with interface first also works correctly."""
        # Act
        self.registry._register_service(DataServiceInterfaceFirst)
        
        # Assert
        service = self.container.resolve(IDataService)
        assert isinstance(service, DataServiceInterfaceFirst)
        
    def test_singleton_lifetime_detected(self):
        """Test that Singleton lifetime is correctly detected."""
        # Act
        self.registry._register_service(LogServiceSingleton)
        
        # Assert
        service1 = self.container.resolve(ILogService)
        service2 = self.container.resolve(ILogService)
        assert service1 is service2  # Should be same instance
        
    def test_scoped_lifetime_detected(self):
        """Test that Scoped lifetime is correctly detected."""
        # Act
        self.registry._register_service(CacheServiceScoped)
        
        # Assert
        scope1 = self.container.create_scope()
        scope2 = self.container.create_scope()
        
        service1a = scope1.resolve(ICacheService)
        service1b = scope1.resolve(ICacheService)
        service2 = scope2.resolve(ICacheService)
        
        assert service1a is service1b  # Same in same scope
        assert service1a is not service2  # Different in different scopes
        
    def test_no_interface_uses_self(self):
        """Test that service without interface registers as itself."""
        # Act
        self.registry._register_service(StandaloneTransient)
        
        # Assert
        service = self.container.resolve(StandaloneTransient)
        assert isinstance(service, StandaloneTransient)
        
    def test_multiple_registrations_different_order(self):
        """Test that multiple services can register to same interface."""
        # Arrange
        class AnotherDataService(Transient, IDataService):
            pass
            
        # Act
        self.registry._register_service(DataServiceTransientFirst)
        # This will overwrite the previous registration
        self.registry._register_service(AnotherDataService)
        
        # Assert
        service = self.container.resolve(IDataService)
        # Should get the last registered one
        assert isinstance(service, AnotherDataService)
        
    def test_scan_module_finds_all_services(self):
        """Test that module scanning finds and registers services correctly."""
        # Act
        # Create a test module mock
        import types
        test_module = types.ModuleType("test_module")
        test_module.Service1 = type("Service1", (Transient, IDataService), {})
        test_module.Service2 = type("Service2", (ILogService, Singleton), {})
        test_module.NotAService = type("NotAService", (object,), {})
        
        count = self.registry._scan_module_members(test_module)
        
        # Assert
        assert count == 2  # Should find 2 services, not NotAService