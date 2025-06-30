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
Dependency injection exceptions.
"""

from typing import Optional, List, Type


class DIError(Exception):
    """
    Base exception for all dependency injection errors.
    
    This is the root exception type for all DI-related failures.
    """
    pass


class ConfigurationError(DIError):
    """
    Exception raised when DI configuration is invalid.
    
    This includes:
    - Module import failures during auto-discovery
    - Invalid service registrations
    - Missing required configuration
    - Bootstrap validation errors
    """
    
    def __init__(
        self,
        message: str,
        module_name: Optional[str] = None,
        service_class: Optional[str] = None,
        suggestion: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.module_name = module_name
        self.service_class = service_class
        self.suggestion = suggestion
        
    def __str__(self) -> str:
        parts = [super().__str__()]
        
        if self.module_name:
            parts.append(f"Module: {self.module_name}")
            
        if self.service_class:
            parts.append(f"Service: {self.service_class}")
            
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
            
        return "\n".join(parts)


class ResolutionError(DIError):
    """
    Exception raised when service resolution fails at runtime.
    
    This includes:
    - Service not registered
    - Dependency injection failures
    - Constructor parameter errors
    """
    
    def __init__(
        self,
        message: str,
        interface: Optional[Type] = None,
        resolution_chain: Optional[List[Type]] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.interface = interface
        self.resolution_chain = resolution_chain or []
        self.original_error = original_error
        
    def __str__(self) -> str:
        parts = [super().__str__()]
        
        if self.interface:
            parts.append(f"Interface: {self.interface.__name__}")
            
        if self.resolution_chain:
            chain_str = " -> ".join(t.__name__ for t in self.resolution_chain)
            parts.append(f"Resolution chain: {chain_str}")
            
        if self.original_error:
            parts.append(f"Original error: {self.original_error}")
            
        return "\n".join(parts)


class CircularDependencyError(DIError):
    """
    Exception raised when circular dependencies are detected.
    
    This prevents infinite recursion during service resolution.
    """
    
    def __init__(
        self,
        message: str,
        cycle_path: Optional[List[Type]] = None,
    ) -> None:
        super().__init__(message)
        self.cycle_path = cycle_path or []
        
    def __str__(self) -> str:
        if self.cycle_path:
            cycle_str = " -> ".join(t.__name__ for t in self.cycle_path)
            return f"{super().__str__()}\nCycle: {cycle_str}"
        return super().__str__()


class ServiceRegistrationError(DIError):
    """
    Exception raised when service registration fails.
    
    This includes:
    - Duplicate registrations
    - Invalid service types
    - Registration validation errors
    """
    
    def __init__(
        self,
        message: str,
        service_type: Optional[Type] = None,
        interface_type: Optional[Type] = None,
    ) -> None:
        super().__init__(message)
        self.service_type = service_type
        self.interface_type = interface_type
        
    def __str__(self) -> str:
        parts = [super().__str__()]
        
        if self.service_type:
            parts.append(f"Service type: {self.service_type.__name__}")
            
        if self.interface_type:
            parts.append(f"Interface type: {self.interface_type.__name__}")
            
        return "\n".join(parts)


class ModuleValidationError(DIError):
    """
    Exception raised when module validation fails during auto-discovery.
    
    This includes:
    - Untrusted module paths
    - Security validation failures
    - Module sandbox violations
    """
    
    def __init__(
        self,
        message: str,
        module_name: str,
        violation_type: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.module_name = module_name
        self.violation_type = violation_type
        
    def __str__(self) -> str:
        parts = [super().__str__()]
        parts.append(f"Module: {self.module_name}")
        
        if self.violation_type:
            parts.append(f"Violation: {self.violation_type}")
            
        return "\n".join(parts)