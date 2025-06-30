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
Tests for module validation.
"""

import pytest

from nautilus_trader.di.module_validator import ModuleValidator
from nautilus_trader.di.exceptions import ModuleValidationError


class TestModuleValidator:
    """Test module validation for secure imports."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ModuleValidator(
            trusted_prefixes=["nautilus_trader", "tests"],
            strict_mode=True,
            audit_imports=False,  # Reduce noise in tests
        )
        
    def test_trusted_module_passes_validation(self):
        """Test that modules under trusted prefixes pass validation."""
        # Act & Assert
        assert self.validator.validate("nautilus_trader.di.container")
        assert self.validator.validate("tests.unit_tests.di.test_module_validator")
        
    def test_untrusted_module_fails_in_strict_mode(self):
        """Test that untrusted modules fail validation in strict mode."""
        # Act & Assert
        with pytest.raises(ModuleValidationError) as exc_info:
            self.validator.validate("untrusted.module")
            
        error = exc_info.value
        assert "not in trusted prefixes" in str(error)
        assert error.module_name == "untrusted.module"
        assert error.violation_type == "untrusted_prefix"
        
    def test_dangerous_patterns_rejected(self):
        """Test that modules with dangerous patterns are rejected."""
        dangerous_modules = [
            "../../../etc/passwd",      # Path traversal
            "/etc/passwd",               # Absolute path
            "C:\\Windows\\System32",     # Windows absolute path
            "nautilus_trader.subprocess", # Subprocess in name
            "eval_module",               # Eval pattern
            "exec_something",            # Exec pattern
        ]
        
        for module_name in dangerous_modules:
            with pytest.raises(ModuleValidationError) as exc_info:
                self.validator.validate(module_name)
                
            error = exc_info.value
            assert "dangerous pattern" in str(error)
            assert error.module_name == module_name
            
    def test_non_strict_mode_logs_warnings(self):
        """Test that non-strict mode logs warnings instead of raising exceptions."""
        # Arrange
        validator = ModuleValidator(
            trusted_prefixes=["nautilus_trader"],
            strict_mode=False,
            audit_imports=False,
        )
        
        # Act & Assert
        # Untrusted module should return False but not raise
        assert not validator.validate("untrusted.module")
        
        # Dangerous pattern should return False but not raise
        assert not validator.validate("../dangerous")
        
    def test_nonexistent_module_handled(self):
        """Test handling of modules that don't exist."""
        # Act & Assert
        with pytest.raises(ModuleValidationError) as exc_info:
            self.validator.validate("nautilus_trader.nonexistent.module")
            
        error = exc_info.value
        assert "does not exist" in str(error)
        assert error.violation_type == "module_not_found"
        
    def test_add_remove_trusted_prefix(self):
        """Test adding and removing trusted prefixes."""
        # Test adding
        self.validator.add_trusted_prefix("new_trusted")
        assert "new_trusted" in self.validator.get_trusted_prefixes()
        assert self.validator.validate("new_trusted.module")
        
        # Test removing
        self.validator.remove_trusted_prefix("new_trusted")
        assert "new_trusted" not in self.validator.get_trusted_prefixes()
        
        with pytest.raises(ModuleValidationError):
            self.validator.validate("new_trusted.module")
            
    def test_strict_mode_toggle(self):
        """Test enabling and disabling strict mode."""
        # Initially strict - should raise
        with pytest.raises(ModuleValidationError):
            self.validator.validate("untrusted.module")
            
        # Disable strict mode - should return False
        self.validator.set_strict_mode(False)
        assert not self.validator.validate("untrusted.module")
        
        # Re-enable strict mode - should raise again
        self.validator.set_strict_mode(True)
        with pytest.raises(ModuleValidationError):
            self.validator.validate("untrusted.module")
            
    def test_existing_python_modules_pass(self):
        """Test that existing Python standard library modules pass basic checks."""
        # Note: These would fail trusted prefix check in strict mode,
        # but should pass the existence and pattern checks
        validator = ModuleValidator(
            trusted_prefixes=["os", "sys", "json"],
            strict_mode=True,
        )
        
        # These should pass validation
        assert validator.validate("os")
        assert validator.validate("sys")
        assert validator.validate("json")
        
    def test_error_message_contains_context(self):
        """Test that error messages contain helpful context."""
        with pytest.raises(ModuleValidationError) as exc_info:
            self.validator.validate("../dangerous")
            
        error_str = str(exc_info.value)
        assert "dangerous pattern" in error_str
        assert "../dangerous" in error_str
        assert "matches pattern" in error_str