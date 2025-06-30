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
Tests for circular dependency detection.
"""

import pytest

from nautilus_trader.di.container import DIContainer, Injectable, Transient
from nautilus_trader.di.exceptions import CircularDependencyError


# Test interfaces and services for circular dependency scenarios
class IServiceA(Injectable):
    """Test service A interface."""
    pass


class IServiceB(Injectable):
    """Test service B interface."""
    pass


class IServiceC(Injectable):
    """Test service C interface."""
    pass


class CircularServiceA(Transient, IServiceA):
    """Service A that depends on B."""
    def __init__(self, service_b: IServiceB):
        self.service_b = service_b


class CircularServiceB(Transient, IServiceB):
    """Service B that depends on A (creates A -> B -> A cycle)."""
    def __init__(self, service_a: IServiceA):
        self.service_a = service_a


class LongChainServiceA(Transient, IServiceA):
    """Service A that depends on B (A -> B -> C -> A cycle)."""
    def __init__(self, service_b: IServiceB):
        self.service_b = service_b


class LongChainServiceB(Transient, IServiceB):
    """Service B that depends on C."""
    def __init__(self, service_c: IServiceC):
        self.service_c = service_c


class LongChainServiceC(Transient, IServiceC):
    """Service C that depends on A (completes the cycle)."""
    def __init__(self, service_a: IServiceA):
        self.service_a = service_a


class SelfDependentService(Transient, IServiceA):
    """Service that depends on itself."""
    def __init__(self, self_ref: IServiceA):
        self.self_ref = self_ref


class ValidServiceA(Transient, IServiceA):
    """Valid service with no circular dependencies."""
    def __init__(self):
        pass


class ValidServiceB(Transient, IServiceB):
    """Valid service that depends on A."""
    def __init__(self, service_a: IServiceA):
        self.service_a = service_a


class TestCircularDependencyDetection:
    """Test circular dependency detection in DIContainer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.container = DIContainer()
        
    def test_simple_circular_dependency_detected(self):
        """Test that simple A -> B -> A cycle is detected."""
        # Arrange
        self.container.register(IServiceA, CircularServiceA)
        self.container.register(IServiceB, CircularServiceB)
        
        # Act & Assert
        with pytest.raises(CircularDependencyError) as exc_info:
            self.container.resolve(IServiceA)
            
        error = exc_info.value
        assert "Circular dependency detected" in str(error)
        assert len(error.cycle_path) >= 2
        
    def test_long_chain_circular_dependency_detected(self):
        """Test that A -> B -> C -> A cycle is detected."""
        # Arrange
        self.container.register(IServiceA, LongChainServiceA)
        self.container.register(IServiceB, LongChainServiceB)
        self.container.register(IServiceC, LongChainServiceC)
        
        # Act & Assert
        with pytest.raises(CircularDependencyError) as exc_info:
            self.container.resolve(IServiceA)
            
        error = exc_info.value
        assert "Circular dependency detected" in str(error)
        assert len(error.cycle_path) >= 3
        
    def test_self_dependency_detected(self):
        """Test that self-dependency is detected."""
        # Arrange
        self.container.register(IServiceA, SelfDependentService)
        
        # Act & Assert
        with pytest.raises(CircularDependencyError) as exc_info:
            self.container.resolve(IServiceA)
            
        error = exc_info.value
        assert "Circular dependency detected" in str(error)
        
    def test_valid_dependencies_work(self):
        """Test that valid non-circular dependencies work correctly."""
        # Arrange
        self.container.register(IServiceA, ValidServiceA)
        self.container.register(IServiceB, ValidServiceB)
        
        # Act
        service_b = self.container.resolve(IServiceB)
        
        # Assert
        assert isinstance(service_b, ValidServiceB)
        assert isinstance(service_b.service_a, ValidServiceA)
        
    def test_multiple_resolution_attempts_after_error(self):
        """Test that container state is properly cleaned up after circular dependency error."""
        # Arrange
        self.container.register(IServiceA, CircularServiceA)
        self.container.register(IServiceB, CircularServiceB)
        
        # Act & Assert - First attempt should fail
        with pytest.raises(CircularDependencyError):
            self.container.resolve(IServiceA)
            
        # Re-register with valid services
        self.container.register(IServiceA, ValidServiceA)
        
        # Second attempt should work
        service = self.container.resolve(IServiceA)
        assert isinstance(service, ValidServiceA)
        
    def test_error_message_contains_cycle_path(self):
        """Test that error message contains the actual cycle path."""
        # Arrange
        self.container.register(IServiceA, CircularServiceA)
        self.container.register(IServiceB, CircularServiceB)
        
        # Act & Assert
        with pytest.raises(CircularDependencyError) as exc_info:
            self.container.resolve(IServiceA)
            
        error_message = str(exc_info.value)
        assert "IServiceA" in error_message
        assert "IServiceB" in error_message
        assert "->" in error_message