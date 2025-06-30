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
Service graph validation for dependency injection container.
"""

import inspect
from typing import Dict, List, Optional, Set, Type, TYPE_CHECKING
from dataclasses import dataclass, field

from nautilus_trader.common.component import Logger
from nautilus_trader.di.exceptions import CircularDependencyError, ResolutionError

if TYPE_CHECKING:
    from nautilus_trader.di.container import DIContainer


@dataclass
class ValidationIssue:
    """Represents an issue found during service graph validation."""
    
    severity: str  # "error", "warning", "info"
    interface: Type
    message: str
    suggestion: Optional[str] = None
    dependency_chain: List[Type] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result of service graph validation."""
    
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    services_validated: int = 0
    
    def has_errors(self) -> bool:
        """Check if there are any error-level issues."""
        return any(issue.severity == "error" for issue in self.issues)
        
    def has_warnings(self) -> bool:
        """Check if there are any warning-level issues."""
        return any(issue.severity == "warning" for issue in self.issues)
        
    def get_errors(self) -> List[ValidationIssue]:
        """Get all error-level issues."""
        return [issue for issue in self.issues if issue.severity == "error"]
        
    def get_warnings(self) -> List[ValidationIssue]:
        """Get all warning-level issues."""
        return [issue for issue in self.issues if issue.severity == "warning"]
        
    def format_report(self) -> str:
        """Format a human-readable validation report."""
        lines = [
            f"Service Graph Validation Report",
            f"==============================",
            f"Services validated: {self.services_validated}",
            f"Overall status: {'PASS' if self.is_valid else 'FAIL'}",
            f"",
        ]
        
        if self.has_errors():
            lines.append("ERRORS:")
            for issue in self.get_errors():
                lines.append(f"  - {issue.interface.__name__}: {issue.message}")
                if issue.suggestion:
                    lines.append(f"    Suggestion: {issue.suggestion}")
            lines.append("")
            
        if self.has_warnings():
            lines.append("WARNINGS:")
            for issue in self.get_warnings():
                lines.append(f"  - {issue.interface.__name__}: {issue.message}")
                if issue.suggestion:
                    lines.append(f"    Suggestion: {issue.suggestion}")
            lines.append("")
            
        return "\n".join(lines)


class ServiceGraphValidator:
    """
    Validates the service dependency graph for completeness and consistency.
    
    Performs "dry run" validation without instantiating services to detect:
    - Missing dependencies
    - Circular dependencies
    - Unresolvable constructor parameters
    - Optional vs required dependency issues
    """
    
    def __init__(self) -> None:
        """Initialize the graph validator."""
        self._logger = Logger(self.__class__.__name__)
        
    def validate(self, container: "DIContainer", partial: bool = False) -> ValidationResult:
        """
        Validate the service dependency graph.
        
        Parameters
        ----------
        container : DIContainer
            Container to validate
        partial : bool
            If True, allows some services to be unresolvable (useful during development)
            
        Returns
        -------
        ValidationResult
            Validation result with issues and statistics
        """
        self._logger.info("Starting service graph validation")
        
        issues = []
        services_validated = 0
        
        # Get all registered services
        registered_services = list(container._services.keys())
        
        for interface in registered_services:
            try:
                self._validate_service(container, interface, set())
                services_validated += 1
                
            except CircularDependencyError as e:
                issues.append(ValidationIssue(
                    severity="error",
                    interface=interface,
                    message=f"Circular dependency detected: {e}",
                    suggestion="Review and break the circular dependency chain",
                    dependency_chain=e.cycle_path,
                ))
                
            except ResolutionError as e:
                severity = "warning" if partial else "error"
                issues.append(ValidationIssue(
                    severity=severity,
                    interface=interface,
                    message=f"Resolution failed: {e}",
                    suggestion="Ensure all dependencies are registered",
                    dependency_chain=e.resolution_chain,
                ))
                
            except Exception as e:
                issues.append(ValidationIssue(
                    severity="error",
                    interface=interface,
                    message=f"Unexpected validation error: {e}",
                    suggestion="Check service implementation and dependencies",
                ))
                
        # Additional validations
        self._check_orphaned_services(container, issues)
        self._check_constructor_complexity(container, issues)
        
        is_valid = not any(issue.severity == "error" for issue in issues)
        
        result = ValidationResult(
            is_valid=is_valid,
            issues=issues,
            services_validated=services_validated,
        )
        
        self._logger.info(
            f"Validation complete: {services_validated} services, "
            f"{len([i for i in issues if i.severity == 'error'])} errors, "
            f"{len([i for i in issues if i.severity == 'warning'])} warnings"
        )
        
        return result
        
    def _validate_service(
        self,
        container: "DIContainer",
        interface: Type,
        resolution_chain: Set[Type],
    ) -> None:
        """
        Validate a single service and its dependencies.
        
        This performs a "dry run" of dependency resolution without
        actually creating instances.
        """
        if interface in resolution_chain:
            cycle_path = list(resolution_chain) + [interface]
            raise CircularDependencyError(
                "Circular dependency detected during validation",
                cycle_path=cycle_path,
            )
            
        resolution_chain.add(interface)
        
        try:
            # Get service descriptor
            if interface not in container._services:
                raise ResolutionError(
                    f"Service {interface.__name__} not registered",
                    interface=interface,
                    resolution_chain=list(resolution_chain),
                )
                
            descriptor = container._services[interface]
            
            # If it has a factory, we can't easily validate dependencies
            if descriptor.factory:
                return
                
            # If it has a pre-created instance, no validation needed
            if descriptor.instance:
                return
                
            # Validate constructor dependencies
            implementation = descriptor.implementation
            if implementation:
                self._validate_constructor(container, implementation, resolution_chain)
                
        finally:
            resolution_chain.remove(interface)
            
    def _validate_constructor(
        self,
        container: "DIContainer",
        cls: Type,
        resolution_chain: Set[Type],
    ) -> None:
        """Validate constructor dependencies for a class."""
        sig = inspect.signature(cls.__init__)
        
        for name, param in sig.parameters.items():
            if name == "self":
                continue
                
            # Skip if parameter has annotation
            if param.annotation == param.empty:
                continue
                
            # Try to validate the dependency
            try:
                self._validate_service(container, param.annotation, resolution_chain.copy())
            except (CircularDependencyError, ResolutionError):
                # If dependency is optional (has default), that's okay
                if param.default == param.empty:
                    raise  # Re-raise for required dependencies
                # Optional dependencies that can't be resolved are okay
                
    def _check_orphaned_services(self, container: "DIContainer", issues: List[ValidationIssue]) -> None:
        """Check for services that are registered but never used."""
        # This is a complex analysis that would require tracking all dependencies
        # For now, we'll implement a simple check
        
        registered_services = set(container._services.keys())
        referenced_services = set()
        
        # Find all services referenced as dependencies
        for interface, descriptor in container._services.items():
            if descriptor.implementation and not descriptor.factory:
                sig = inspect.signature(descriptor.implementation.__init__)
                for param in sig.parameters.values():
                    if param.annotation != param.empty and param.annotation in registered_services:
                        referenced_services.add(param.annotation)
                        
        # Services that are registered but never referenced might be orphaned
        potentially_orphaned = registered_services - referenced_services
        
        for interface in potentially_orphaned:
            # Don't flag root services as orphaned (these are likely entry points)
            if self._is_likely_root_service(interface):
                continue
                
            issues.append(ValidationIssue(
                severity="info",
                interface=interface,
                message="Service appears to be orphaned (not referenced by other services)",
                suggestion="Consider if this service is still needed or if it should be a root service",
            ))
            
    def _check_constructor_complexity(self, container: "DIContainer", issues: List[ValidationIssue]) -> None:
        """Check for constructors with too many dependencies."""
        # Get threshold from container config or use default
        max_dependencies = 10  # Default
        if hasattr(container, '_config') and container._config:
            max_dependencies = container._config.validation.max_constructor_dependencies
        
        for interface, descriptor in container._services.items():
            if descriptor.implementation and not descriptor.factory:
                sig = inspect.signature(descriptor.implementation.__init__)
                param_count = len([p for p in sig.parameters.values() if p.name != "self"])
                
                if param_count > max_dependencies:
                    issues.append(ValidationIssue(
                        severity="warning",
                        interface=interface,
                        message=f"Constructor has {param_count} dependencies (max recommended: {max_dependencies})",
                        suggestion="Consider refactoring to reduce dependencies or use factory pattern",
                    ))
                    
    def _is_likely_root_service(self, interface: Type) -> bool:
        """Check if a service is likely a root service (entry point)."""
        # Heuristic: services with names containing certain patterns
        # are likely to be root services
        name = interface.__name__.lower()
        root_patterns = ["engine", "manager", "service", "controller", "handler", "processor"]
        
        # Use configured patterns if available
        if hasattr(self, '_container') and hasattr(self._container, '_config') and self._container._config:
            root_patterns = self._container._config.validation.root_service_patterns
        
        return any(pattern in name for pattern in root_patterns)