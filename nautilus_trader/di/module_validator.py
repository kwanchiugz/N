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
Module validation for secure dynamic imports.
"""

import importlib.util
import os
import re
from pathlib import Path
from typing import List, Optional, Set

from nautilus_trader.common.component import Logger
from nautilus_trader.di.exceptions import ModuleValidationError


class ModuleValidator:
    """
    Validator for dynamic module imports.
    
    Provides security controls for module loading during auto-discovery:
    - Whitelist-based module path validation
    - Detection of path traversal attempts
    - Validation against dangerous module patterns
    - Audit logging of import attempts
    """
    
    def __init__(
        self,
        trusted_prefixes: Optional[List[str]] = None,
        strict_mode: bool = True,
        audit_imports: bool = True,
    ) -> None:
        """
        Initialize module validator.
        
        Parameters
        ----------
        trusted_prefixes : List[str], optional
            List of trusted module path prefixes. Defaults to ["nautilus_trader"]
        strict_mode : bool
            If True, only allows modules matching trusted prefixes
        audit_imports : bool
            If True, logs all import attempts for security auditing
        """
        self._trusted_prefixes = trusted_prefixes or ["nautilus_trader"]
        self._strict_mode = strict_mode
        self._audit_imports = audit_imports
        self._logger = Logger(self.__class__.__name__)
        
        # Dangerous patterns that should never be imported
        self._dangerous_patterns = {
            r".*\.\.(\/|\\).*",           # Path traversal
            r"^\/.*",                     # Absolute paths
            r"^[A-Z]:\\.*",              # Windows absolute paths
            r".*subprocess.*",            # Process execution
            r".*os\.system.*",           # System commands
            r".*eval.*",                 # Code evaluation
            r".*exec.*",                 # Code execution
            r".*__import__.*",           # Dynamic imports
        }
        
        # Compile patterns for efficiency
        self._compiled_patterns = [re.compile(pattern) for pattern in self._dangerous_patterns]
        
    def validate(self, module_name: str) -> bool:
        """
        Validate if a module can be safely imported.
        
        Parameters
        ----------
        module_name : str
            Name of the module to validate
            
        Returns
        -------
        bool
            True if module is safe to import
            
        Raises
        ------
        ModuleValidationError
            If module fails validation in strict mode
        """
        if self._audit_imports:
            self._logger.debug(f"Validating module import: {module_name}")
            
        # Check for dangerous patterns
        violation = self._check_dangerous_patterns(module_name)
        if violation:
            if self._strict_mode:
                raise ModuleValidationError(
                    f"Module '{module_name}' contains dangerous pattern",
                    module_name=module_name,
                    violation_type=violation,
                )
            else:
                self._logger.warning(f"Dangerous pattern in module {module_name}: {violation}")
                return False
                
        # Check trusted prefixes
        if self._strict_mode and not self._is_trusted_module(module_name):
            raise ModuleValidationError(
                f"Module '{module_name}' not in trusted prefixes: {self._trusted_prefixes}",
                module_name=module_name,
                violation_type="untrusted_prefix",
            )
            
        # Check if module exists and is accessible
        if not self._module_exists(module_name):
            if self._strict_mode:
                raise ModuleValidationError(
                    f"Module '{module_name}' does not exist or is not accessible",
                    module_name=module_name,
                    violation_type="module_not_found",
                )
            else:
                return False
                
        if self._audit_imports:
            self._logger.info(f"Module validation passed: {module_name}")
            
        return True
        
    def _check_dangerous_patterns(self, module_name: str) -> Optional[str]:
        """Check if module name matches any dangerous patterns."""
        for pattern in self._compiled_patterns:
            if pattern.match(module_name):
                return f"matches pattern: {pattern.pattern}"
        return None
        
    def _is_trusted_module(self, module_name: str) -> bool:
        """Check if module is under a trusted prefix."""
        return any(module_name.startswith(prefix) for prefix in self._trusted_prefixes)
        
    def _module_exists(self, module_name: str) -> bool:
        """Check if module exists and can be imported."""
        try:
            spec = importlib.util.find_spec(module_name)
            return spec is not None
        except (ImportError, ValueError, ModuleNotFoundError):
            return False
            
    def add_trusted_prefix(self, prefix: str) -> None:
        """
        Add a new trusted module prefix.
        
        Parameters
        ----------
        prefix : str
            Module prefix to trust (e.g., "myapp.services")
        """
        if prefix not in self._trusted_prefixes:
            self._trusted_prefixes.append(prefix)
            self._logger.info(f"Added trusted prefix: {prefix}")
            
    def remove_trusted_prefix(self, prefix: str) -> None:
        """
        Remove a trusted module prefix.
        
        Parameters
        ----------
        prefix : str
            Module prefix to remove
        """
        if prefix in self._trusted_prefixes:
            self._trusted_prefixes.remove(prefix)
            self._logger.info(f"Removed trusted prefix: {prefix}")
            
    def get_trusted_prefixes(self) -> List[str]:
        """
        Get list of trusted module prefixes.
        
        Returns
        -------
        List[str]
            Current trusted prefixes
        """
        return self._trusted_prefixes.copy()
        
    def set_strict_mode(self, enabled: bool) -> None:
        """
        Enable or disable strict mode.
        
        Parameters
        ----------
        enabled : bool
            If True, validation failures raise exceptions
        """
        self._strict_mode = enabled
        self._logger.info(f"Strict mode {'enabled' if enabled else 'disabled'}")


# Default validator instance
_default_validator = ModuleValidator()


def get_module_validator() -> ModuleValidator:
    """Get the default module validator instance."""
    return _default_validator


def set_trusted_prefixes(prefixes: List[str]) -> None:
    """
    Set trusted module prefixes for the default validator.
    
    Parameters
    ----------
    prefixes : List[str]
        List of trusted module prefixes
    """
    _default_validator._trusted_prefixes = prefixes


def validate_module(module_name: str) -> bool:
    """
    Validate a module using the default validator.
    
    Parameters
    ----------
    module_name : str
        Module name to validate
        
    Returns
    -------
    bool
        True if module is safe to import
    """
    return _default_validator.validate(module_name)