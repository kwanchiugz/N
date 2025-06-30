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
Integration tests for all DI system fixes.
"""

import pytest

from nautilus_trader.di.container import DIContainer, Injectable, Singleton, Transient
from nautilus_trader.di.registry import ServiceRegistry
from nautilus_trader.di.bootstrap import Bootstrap
from nautilus_trader.di.exceptions import (
    CircularDependencyError,
    ConfigurationError,
    ModuleValidationError,
    ServiceRegistrationError,
)
from nautilus_trader.di.module_validator import ModuleValidator
from nautilus_trader.di.graph_validator import ServiceGraphValidator


# Test services for integration testing
class IDataService(Injectable):
    """Test data service interface."""
    pass


class ILogService(Injectable):
    """Test log service interface."""
    pass


class ICacheService(Injectable):
    """Test cache service interface."""
    pass


class DataService(Transient, IDataService):
    """Data service implementation."""
    def __init__(self, log_service: ILogService):
        self.log_service = log_service


class LogService(Singleton, ILogService):
    """Log service implementation."""
    def __init__(self):
        pass


class CacheService(Transient, ICacheService):
    """Cache service implementation."""
    def __init__(self, data_service: IDataService, log_service: ILogService):
        self.data_service = data_service
        self.log_service = log_service


class TestDIIntegration:
    """Integration tests for all DI system improvements."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.container = DIContainer()
        self.registry = ServiceRegistry(self.container)
        
    def test_interface_detection_fix_works(self):
        """Test that the interface detection fix works correctly."""
        # Register services with lifetime first
        self.registry._register_service(DataService)  # Transient, IDataService
        self.registry._register_service(LogService)   # Singleton, ILogService
        
        # Should be able to resolve by interface, not by lifetime
        log_service = self.container.resolve(ILogService)
        assert isinstance(log_service, LogService)
        
        data_service = self.container.resolve(IDataService)
        assert isinstance(data_service, DataService)
        assert isinstance(data_service.log_service, LogService)
        
    def test_circular_dependency_detection_works(self):
        """Test that circular dependency detection prevents infinite loops."""
        # Create circular dependency: A -> B -> A
        class CircularA(Transient, IDataService):
            def __init__(self, service_b: ILogService):
                self.service_b = service_b
                
        class CircularB(Transient, ILogService):
            def __init__(self, service_a: IDataService):
                self.service_a = service_a
                
        self.container.register(IDataService, CircularA)
        self.container.register(ILogService, CircularB)
        
        with pytest.raises(CircularDependencyError) as exc_info:
            self.container.resolve(IDataService)
            
        error = exc_info.value
        assert "Circular dependency detected" in str(error)
        
    def test_fail_fast_error_handling(self):
        """Test that errors are properly propagated instead of being silently logged."""
        # Test service registration failure
        with pytest.raises(ServiceRegistrationError):
            # Try to register with invalid configuration that should fail
            self.registry._register_service(type("InvalidService", (), {}))
            
    def test_module_validation_security(self):
        """Test that module validation prevents dangerous imports."""
        validator = ModuleValidator(
            trusted_prefixes=["nautilus_trader"],
            strict_mode=True,
        )
        
        # Should reject dangerous module names
        with pytest.raises(ModuleValidationError):
            validator.validate("../../../etc/passwd")
            
        with pytest.raises(ModuleValidationError):
            validator.validate("untrusted.module")
            
        # Should allow trusted modules
        assert validator.validate("nautilus_trader.di.container")
        
    def test_service_graph_validation(self):
        """Test that service graph validation detects issues."""
        # Register valid services
        self.container.register(ILogService, LogService)
        self.container.register(IDataService, DataService)
        self.container.register(ICacheService, CacheService)
        
        validator = ServiceGraphValidator()
        result = validator.validate(self.container)
        
        assert result.is_valid
        assert result.services_validated == 3
        
    def test_complete_bootstrap_workflow(self):
        """Test the complete bootstrap workflow with all fixes."""
        # Create bootstrap with security configured
        module_validator = ModuleValidator(
            trusted_prefixes=["nautilus_trader", "tests"],
            strict_mode=True,
        )
        
        bootstrap = Bootstrap(
            container=self.container,
            module_validator=module_validator,
        )
        
        # Configure services
        bootstrap.configure_core_services()
        
        # Register test services manually (since auto-discovery would need real modules)
        self.container.register(ILogService, LogService)
        self.container.register(IDataService, DataService)
        self.container.register(ICacheService, CacheService)
        
        # Validate everything
        result = bootstrap.validate(validate_graph=True)
        assert result is True
        
        # Test that services work correctly
        cache_service = self.container.resolve(ICacheService)
        assert isinstance(cache_service, CacheService)
        assert isinstance(cache_service.data_service, DataService)
        assert isinstance(cache_service.log_service, LogService)
        
        # Verify singleton behavior
        log1 = self.container.resolve(ILogService)
        log2 = self.container.resolve(ILogService)
        assert log1 is log2  # Should be same instance
        
        # Verify transient behavior
        data1 = self.container.resolve(IDataService)
        data2 = self.container.resolve(IDataService)
        assert data1 is not data2  # Should be different instances
        
    def test_error_messages_are_informative(self):
        """Test that error messages contain helpful information."""
        # Test circular dependency error
        class CircularA(Transient, IDataService):
            def __init__(self, service: ILogService):
                pass
                
        class CircularB(Transient, ILogService):
            def __init__(self, service: IDataService):
                pass
                
        self.container.register(IDataService, CircularA)
        self.container.register(ILogService, CircularB)
        
        with pytest.raises(CircularDependencyError) as exc_info:
            self.container.resolve(IDataService)
            
        error_message = str(exc_info.value)
        assert "IDataService" in error_message
        assert "ILogService" in error_message
        assert "->" in error_message
        
    def test_module_validator_integration_with_registry(self):
        """Test that registry properly uses module validator."""
        validator = ModuleValidator(
            trusted_prefixes=["nautilus_trader"],
            strict_mode=True,
        )
        
        registry = ServiceRegistry(self.container, validator)
        
        # Should reject untrusted modules
        with pytest.raises(ConfigurationError) as exc_info:
            registry.scan_module("untrusted.module")
            
        assert "Module validation failed" in str(exc_info.value)
        
    def test_backward_compatibility(self):
        """Test that manual registration still works as before."""
        # Manual registration should work without any auto-discovery
        self.container.register_singleton(ILogService, LogService)
        self.container.register_transient(IDataService, DataService)
        
        # Should work exactly as before
        log_service = self.container.resolve(ILogService)
        data_service = self.container.resolve(IDataService)
        
        assert isinstance(log_service, LogService)
        assert isinstance(data_service, DataService)
        assert isinstance(data_service.log_service, LogService)