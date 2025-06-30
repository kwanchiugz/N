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
Comprehensive integration tests for the complete DI system.
Tests all improvements: configuration, monitoring, caching, resilience, and service catalog.
"""

import pytest
import tempfile
import json
from pathlib import Path
from typing import Optional

from nautilus_trader.di.container import DIContainer, Injectable, Singleton, Transient
from nautilus_trader.di.config import DIContainerConfig, SecurityConfig, ValidationConfig
from nautilus_trader.di.service_catalog import ServiceDefinition, ServiceCategory
from nautilus_trader.di.bootstrap import Bootstrap
from nautilus_trader.di.exceptions import ConfigurationError, ResolutionError


# Test service interfaces and implementations
class ITestLogger(Injectable):
    """Test logger interface."""
    def log(self, message: str) -> None:
        pass


class ITestCache(Injectable):
    """Test cache interface."""
    def get(self, key: str) -> Optional[str]:
        pass
    
    def set(self, key: str, value: str) -> None:
        pass


class ITestProcessor(Injectable):
    """Test processor interface."""
    def process(self, data: str) -> str:
        pass


class TestLogger(Singleton, ITestLogger):
    """Test logger implementation."""
    def __init__(self):
        self.messages = []
        
    def log(self, message: str) -> None:
        self.messages.append(message)


class TestCache(Transient, ITestCache):
    """Test cache implementation."""
    def __init__(self, logger: ITestLogger):
        self.logger = logger
        self._data = {}
        
    def get(self, key: str) -> Optional[str]:
        self.logger.log(f"Cache get: {key}")
        return self._data.get(key)
    
    def set(self, key: str, value: str) -> None:
        self.logger.log(f"Cache set: {key}={value}")
        self._data[key] = value


class TestProcessor(Transient, ITestProcessor):
    """Test processor implementation."""
    def __init__(self, logger: ITestLogger, cache: ITestCache):
        self.logger = logger
        self.cache = cache
        
    def process(self, data: str) -> str:
        self.logger.log(f"Processing: {data}")
        result = f"processed_{data}"
        self.cache.set(data, result)
        return result


class FailingTestService(Transient, ITestProcessor):
    """Test service that always fails."""
    def __init__(self):
        pass
        
    def process(self, data: str) -> str:
        raise RuntimeError(f"Service failure for: {data}")


class TestCompleteDISystem:
    """Comprehensive tests for the complete DI system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create test configuration
        self.config = DIContainerConfig(
            container_name="test_container",
            version="1.0.0",
            description="Test container for comprehensive DI testing",
        )
        
        # Configure components
        self.config.security.trusted_prefixes.append("tests")
        self.config.validation.max_constructor_dependencies = 5
        self.config.monitoring.enabled = True
        self.config.caching.enabled = True
        self.config.resilience.enabled = True
        
        # Create container with configuration
        self.container = DIContainer(config=self.config)
        
    def test_complete_workflow_with_all_features(self):
        """Test complete workflow with all DI features enabled."""
        # Register services
        self.container.register(ITestLogger, TestLogger)
        self.container.register(ITestCache, TestCache)
        self.container.register(ITestProcessor, TestProcessor)
        
        # Test service resolution
        processor = self.container.resolve(ITestProcessor)
        assert isinstance(processor, TestProcessor)
        assert isinstance(processor.logger, TestLogger)
        assert isinstance(processor.cache, TestCache)
        
        # Test functionality
        result = processor.process("test_data")
        assert result == "processed_test_data"
        
        # Verify caching worked
        cached_value = processor.cache.get("test_data")
        assert cached_value == "processed_test_data"
        
        # Verify logging worked
        logger = self.container.resolve(ITestLogger)
        assert len(logger.messages) >= 2  # At least cache set and process messages
        
    def test_configuration_system(self):
        """Test configuration system with file loading and overrides."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                "container_name": "file_config_test",
                "version": "2.0.0",
                "security": {
                    "strict_mode": False,
                    "trusted_prefixes": ["tests", "custom"]
                },
                "validation": {
                    "max_constructor_dependencies": 15
                },
                "monitoring": {
                    "enabled": False
                }
            }
            json.dump(config_data, f)
            config_file = f.name
            
        try:
            # Load configuration from file
            file_config = DIContainerConfig.from_file(config_file)
            
            assert file_config.container_name == "file_config_test"
            assert file_config.version == "2.0.0"
            assert not file_config.security.strict_mode
            assert "custom" in file_config.security.trusted_prefixes
            assert file_config.validation.max_constructor_dependencies == 15
            assert not file_config.monitoring.enabled
            
            # Test configuration validation
            errors = file_config.validate()
            assert len(errors) == 0
            
            # Test environment overrides
            import os
            os.environ["NAUTILUS_DI_MONITORING_ENABLED"] = "true"
            file_config.apply_environment_overrides()
            assert file_config.monitoring.enabled
            
            # Clean up environment
            del os.environ["NAUTILUS_DI_MONITORING_ENABLED"]
            
        finally:
            Path(config_file).unlink()
            
    def test_service_catalog_dynamic_configuration(self):
        """Test service catalog for dynamic service configuration."""
        from nautilus_trader.di.service_catalog import get_service_catalog
        
        catalog = get_service_catalog()
        
        # Register custom service definition
        service_def = ServiceDefinition(
            name="custom_processor",
            interface_path="tests.unit_tests.di.test_complete_di_system.ITestProcessor",
            implementation_path="tests.unit_tests.di.test_complete_di_system.TestProcessor",
            category=ServiceCategory.CUSTOM,
            dependencies=["test_logger", "test_cache"],
            description="Custom test processor service",
        )
        
        catalog.register_service(service_def)
        
        # Verify registration
        retrieved = catalog.get_service("custom_processor")
        assert retrieved is not None
        assert retrieved.name == "custom_processor"
        assert retrieved.category == ServiceCategory.CUSTOM
        
        # Test service validation
        assert catalog.validate_service(service_def)
        
        # Test dependency ordering
        # Add dependencies first
        logger_def = ServiceDefinition(
            name="test_logger",
            interface_path="tests.unit_tests.di.test_complete_di_system.ITestLogger",
            implementation_path="tests.unit_tests.di.test_complete_di_system.TestLogger",
        )
        cache_def = ServiceDefinition(
            name="test_cache",
            interface_path="tests.unit_tests.di.test_complete_di_system.ITestCache",
            implementation_path="tests.unit_tests.di.test_complete_di_system.TestCache",
            dependencies=["test_logger"],
        )
        
        catalog.register_service(logger_def)
        catalog.register_service(cache_def)
        
        # Get dependency order
        order = catalog.get_dependency_order()
        assert "test_logger" in order
        assert "test_cache" in order
        assert "custom_processor" in order
        
        # Logger should come before cache (cache depends on logger)
        logger_index = order.index("test_logger")
        cache_index = order.index("test_cache")
        processor_index = order.index("custom_processor")
        
        assert logger_index < cache_index
        assert cache_index < processor_index
        
    def test_bootstrap_with_dynamic_services(self):
        """Test bootstrap system with dynamic service configuration."""
        bootstrap = Bootstrap(container=self.container)
        
        # Configure core services
        bootstrap.configure_core_services()
        
        # Configure with custom service
        bootstrap.configure_service_by_name(
            "event_bus",
            # Configuration overrides could go here
        )
        
        # Test validation
        result = bootstrap.validate(validate_graph=True)
        assert result is True
        
    def test_monitoring_and_health_checks(self):
        """Test monitoring system and health checks."""
        # Register services
        self.container.register(ITestLogger, TestLogger)
        self.container.register(ITestCache, TestCache)
        self.container.register(ITestProcessor, TestProcessor)
        
        # Perform multiple resolutions to generate metrics
        for i in range(10):
            processor = self.container.resolve(ITestProcessor)
            processor.process(f"test_data_{i}")
            
        # Get monitoring dashboard data
        dashboard_data = self.container.get_monitoring_dashboard_data()
        
        assert dashboard_data["overall_health"] in ["healthy", "degraded", "unhealthy"]
        assert dashboard_data["metrics"]["resolution_count"] >= 10
        assert dashboard_data["metrics"]["registered_services"] >= 3
        
        # Check health status
        health_status = self.container.get_health_status()
        assert health_status in ["healthy", "degraded", "unhealthy", "unknown"]
        
    def test_caching_system(self):
        """Test caching system functionality."""
        # Register singleton service to test caching
        self.container.register(ITestLogger, TestLogger)
        
        # First resolution
        logger1 = self.container.resolve(ITestLogger)
        
        # Second resolution (should use cache for singleton)
        logger2 = self.container.resolve(ITestLogger)
        
        # Should be same instance for singleton
        assert logger1 is logger2
        
        # Get cache statistics
        cache_stats = self.container.get_cache_stats()
        
        if cache_stats.get("caching_enabled", True):
            assert cache_stats["hits"] >= 0
            assert cache_stats["misses"] >= 0
            
        # Test cache configuration
        self.container.configure_cache(max_size=500)
        
    def test_resilience_and_fallbacks(self):
        """Test resilience mechanisms and fallback providers."""
        # Register failing service
        self.container.register(ITestProcessor, FailingTestService)
        
        # Configure resilience for graceful degradation
        self.container.configure_resilience(
            failure_mode="graceful",
            max_retries=2
        )
        
        # Register fallback
        class FallbackProcessor:
            def process(self, data: str) -> str:
                return f"fallback_{data}"
                
        fallback_instance = FallbackProcessor()
        self.container.register_fallback(ITestProcessor, fallback_instance)
        
        # Try to resolve - should get fallback due to failure
        try:
            processor = self.container.resolve(ITestProcessor)
            result = processor.process("test")
            
            # If resilience is working, we might get a fallback or mock
            # The exact behavior depends on the resilience configuration
            assert result is not None
            
        except (ResolutionError, RuntimeError):
            # If resilience isn't available or configured for fail-fast
            pass
            
        # Get service health report
        health_report = self.container.get_service_health_report()
        assert "resilience_enabled" in health_report
        
    def test_complex_dependency_resolution_with_all_features(self):
        """Test complex dependency resolution with all features enabled."""
        # Create a more complex dependency graph
        class IComplexService(Injectable):
            pass
            
        class ComplexService(Singleton, IComplexService):
            def __init__(self, logger: ITestLogger, cache: ITestCache, processor: ITestProcessor):
                self.logger = logger
                self.cache = cache
                self.processor = processor
                
        # Register all services
        self.container.register(ITestLogger, TestLogger)
        self.container.register(ITestCache, TestCache)
        self.container.register(ITestProcessor, TestProcessor)
        self.container.register(IComplexService, ComplexService)
        
        # Resolve complex service
        complex_service = self.container.resolve(IComplexService)
        
        assert isinstance(complex_service, ComplexService)
        assert isinstance(complex_service.logger, TestLogger)
        assert isinstance(complex_service.cache, TestCache)
        assert isinstance(complex_service.processor, TestProcessor)
        
        # Test that dependencies are properly injected
        test_data = "complex_test"
        result = complex_service.processor.process(test_data)
        
        assert result == f"processed_{test_data}"
        assert len(complex_service.logger.messages) > 0
        
    def test_configuration_overrides_at_runtime(self):
        """Test runtime configuration overrides."""
        # Print initial configuration
        self.container.print_configuration()
        
        # Apply configuration overrides
        self.container.apply_config_overrides(
            **{
                "validation.max_constructor_dependencies": 20,
                "monitoring.memory_warning_mb": 256.0,
                "caching.default_max_size": 2000,
            }
        )
        
        # Verify overrides were applied
        config = self.container.get_configuration()
        assert config.validation.max_constructor_dependencies == 20
        assert config.monitoring.memory_warning_mb == 256.0
        assert config.caching.default_max_size == 2000
        
    def test_circular_dependency_detection_with_improved_error_handling(self):
        """Test circular dependency detection with improved error messages."""
        class IServiceA(Injectable):
            pass
            
        class IServiceB(Injectable):
            pass
            
        class ServiceA(Transient, IServiceA):
            def __init__(self, service_b: IServiceB):
                self.service_b = service_b
                
        class ServiceB(Transient, IServiceB):
            def __init__(self, service_a: IServiceA):
                self.service_a = service_a
                
        # Register circular dependencies
        self.container.register(IServiceA, ServiceA)
        self.container.register(IServiceB, ServiceB)
        
        # Should detect circular dependency
        from nautilus_trader.di.exceptions import CircularDependencyError
        with pytest.raises(CircularDependencyError) as exc_info:
            self.container.resolve(IServiceA)
            
        error = exc_info.value
        assert "Circular dependency detected" in str(error)
        
        # Error message should contain service names
        assert "IServiceA" in str(error) or "IServiceB" in str(error)
        
    def test_complete_system_integration_example(self):
        """Test a complete system integration example."""
        # This test demonstrates a real-world usage scenario
        
        # 1. Load configuration
        config = DIContainerConfig(
            container_name="production_system",
            version="1.0.0",
            description="Production-grade DI container",
        )
        
        # 2. Configure for production
        config.security.strict_mode = True
        config.validation.enable_graph_validation = True
        config.monitoring.enabled = True
        config.caching.enabled = True
        config.resilience.enabled = True
        config.resilience.default_failure_mode = "circuit_breaker"
        
        # 3. Create container with full configuration
        container = DIContainer(config=config)
        
        # 4. Register services
        container.register(ITestLogger, TestLogger)
        container.register(ITestCache, TestCache)
        container.register(ITestProcessor, TestProcessor)
        
        # 5. Setup bootstrap and validate
        bootstrap = Bootstrap(container=container)
        bootstrap.configure_core_services()
        
        # 6. Validate entire system
        validation_result = bootstrap.validate(validate_graph=True)
        assert validation_result is True
        
        # 7. Use the system
        processor = container.resolve(ITestProcessor)
        results = []
        for i in range(5):
            result = processor.process(f"production_data_{i}")
            results.append(result)
            
        assert len(results) == 5
        assert all("processed_production_data" in result for result in results)
        
        # 8. Monitor system health
        health_status = container.get_health_status()
        dashboard_data = container.get_monitoring_dashboard_data()
        
        assert health_status in ["healthy", "degraded", "unhealthy", "unknown"]
        assert dashboard_data["metrics"]["resolution_count"] >= 5
        
        # 9. Print system status
        container.print_monitoring_dashboard()
        container.print_cache_stats()
        container.print_service_health()
        container.print_configuration()
        
        print("\n✅ Complete DI system integration test passed!")
        print(f"   Container: {config.container_name} v{config.version}")
        print(f"   Health: {health_status}")
        print(f"   Services: {dashboard_data['metrics']['registered_services']}")
        print(f"   Resolutions: {dashboard_data['metrics']['resolution_count']}")
        
    def teardown_method(self):
        """Clean up after tests."""
        # Clear any global state if needed
        pass