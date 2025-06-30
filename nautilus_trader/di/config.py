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
Comprehensive configuration system for dependency injection container.
"""

import os
import json
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from nautilus_trader.common.component import Logger
from nautilus_trader.di.exceptions import ConfigurationError


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class CacheStrategy(str, Enum):
    """Cache strategies."""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    WEAK = "weak"
    NONE = "none"


class FailureMode(str, Enum):
    """Service failure modes."""
    FAIL_FAST = "fail_fast"
    GRACEFUL_DEGRADATION = "graceful"
    CIRCUIT_BREAKER = "circuit_breaker"
    RETRY_WITH_BACKOFF = "retry_backoff"
    DEFAULT_INSTANCE = "default_instance"


@dataclass
class SecurityConfig:
    """Security configuration for DI container."""
    
    # Module validation settings
    trusted_prefixes: List[str] = field(default_factory=lambda: ["nautilus_trader"])
    strict_mode: bool = True
    audit_imports: bool = True
    
    # Dangerous patterns (regex patterns to block)
    dangerous_patterns: List[str] = field(default_factory=lambda: [
        r".*\.\.(\/|\\).*",           # Path traversal
        r"^\/.*",                     # Absolute paths
        r"^[A-Z]:\\.*",              # Windows absolute paths
        r".*subprocess.*",            # Process execution
        r".*os\.system.*",           # System commands
        r".*eval.*",                 # Code evaluation
        r".*exec.*",                 # Code execution
        r".*__import__.*",           # Dynamic imports
    ])
    
    # Security thresholds
    max_import_depth: int = 10
    max_module_scan_time: float = 30.0  # seconds
    
    def add_trusted_prefix(self, prefix: str) -> None:
        """Add trusted module prefix."""
        if prefix not in self.trusted_prefixes:
            self.trusted_prefixes.append(prefix)
            
    def add_dangerous_pattern(self, pattern: str) -> None:
        """Add dangerous pattern to block."""
        if pattern not in self.dangerous_patterns:
            self.dangerous_patterns.append(pattern)


@dataclass
class ValidationConfig:
    """Validation configuration for DI container."""
    
    # Graph validation settings
    enable_graph_validation: bool = True
    enable_startup_validation: bool = True
    partial_validation_mode: bool = False
    
    # Constructor complexity limits
    max_constructor_dependencies: int = 10
    max_dependency_depth: int = 20
    
    # Circular dependency detection
    enable_circular_detection: bool = True
    max_resolution_chain_length: int = 50
    
    # Orphaned service detection
    enable_orphaned_detection: bool = True
    root_service_patterns: List[str] = field(default_factory=lambda: [
        "engine", "manager", "service", "controller", "handler", "processor"
    ])
    
    # Performance thresholds
    max_validation_time: float = 60.0  # seconds
    validation_timeout: float = 5.0    # per service
    
    def add_root_service_pattern(self, pattern: str) -> None:
        """Add root service pattern."""
        if pattern not in self.root_service_patterns:
            self.root_service_patterns.append(pattern)


@dataclass
class CacheConfig:
    """Cache configuration for DI container."""
    
    # General cache settings
    enabled: bool = True
    default_strategy: CacheStrategy = CacheStrategy.LRU
    default_max_size: int = 1000
    default_ttl_seconds: Optional[float] = 300.0  # 5 minutes
    
    # Cache behavior
    enable_weak_references: bool = True
    cleanup_interval: float = 60.0  # seconds
    auto_cleanup: bool = True
    
    # Service-specific cache settings
    cache_singletons: bool = True
    cache_scoped: bool = True
    cache_transients: bool = False
    
    # Performance settings
    max_cache_memory_mb: Optional[float] = None
    eviction_batch_size: int = 10


@dataclass
class MonitoringConfig:
    """Monitoring configuration for DI container."""
    
    # General monitoring
    enabled: bool = True
    collect_metrics: bool = True
    enable_health_checks: bool = True
    
    # Metrics collection
    metrics_history_size: int = 1000
    collect_system_metrics: bool = True
    collect_container_metrics: bool = True
    
    # Health check settings
    health_check_interval: float = 10.0  # seconds
    health_check_timeout: float = 5.0
    
    # Performance monitoring
    track_resolution_times: bool = True
    track_cache_performance: bool = True
    track_error_rates: bool = True
    
    # Alerting thresholds
    memory_warning_mb: float = 500.0
    memory_critical_mb: float = 1000.0
    error_rate_warning: float = 0.05  # 5%
    error_rate_critical: float = 0.10  # 10%
    avg_resolution_time_warning: float = 100.0  # ms
    avg_resolution_time_critical: float = 500.0  # ms


@dataclass
class ResilienceConfig:
    """Resilience configuration for DI container."""
    
    # General resilience
    enabled: bool = True
    default_failure_mode: FailureMode = FailureMode.GRACEFUL_DEGRADATION
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    backoff_multiplier: float = 2.0
    max_retry_delay: float = 30.0
    
    # Circuit breaker settings
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 30.0
    circuit_breaker_recovery_timeout: float = 5.0
    
    # Health monitoring
    health_check_interval: float = 10.0
    degradation_timeout: float = 300.0  # 5 minutes
    failure_history_size: int = 100
    
    # Fallback settings
    enable_mock_fallbacks: bool = True
    enable_default_fallbacks: bool = True
    fallback_timeout: float = 5.0


@dataclass
class PerformanceConfig:
    """Performance configuration for DI container."""
    
    # General performance
    enable_async_resolution: bool = False
    enable_lazy_loading: bool = True
    enable_object_pooling: bool = False
    
    # Threading
    max_concurrent_resolutions: int = 100
    thread_pool_size: Optional[int] = None
    
    # Memory management
    enable_garbage_collection: bool = True
    gc_threshold: int = 1000
    
    # Optimization
    enable_jit_compilation: bool = False
    enable_profile_guided_optimization: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration for DI container."""
    
    # General logging
    level: LogLevel = LogLevel.INFO
    enable_structured_logging: bool = False
    
    # Log targets
    log_to_console: bool = True
    log_to_file: bool = False
    log_file_path: Optional[str] = None
    
    # Log content
    log_service_registrations: bool = True
    log_service_resolutions: bool = False
    log_performance_metrics: bool = False
    log_security_events: bool = True
    log_errors_only: bool = False
    
    # Log formatting
    include_timestamps: bool = True
    include_thread_info: bool = False
    include_stack_traces: bool = True


@dataclass
class DIContainerConfig:
    """Complete configuration for DI container."""
    
    # Core settings
    container_name: str = "default"
    version: str = "1.0.0"
    description: Optional[str] = None
    
    # Component configurations
    security: SecurityConfig = field(default_factory=SecurityConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    caching: CacheConfig = field(default_factory=CacheConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    resilience: ResilienceConfig = field(default_factory=ResilienceConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Service configurations
    services: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Environment overrides
    enable_env_overrides: bool = True
    env_prefix: str = "NAUTILUS_DI_"
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "DIContainerConfig":
        """
        Load configuration from file.
        
        Parameters
        ----------
        config_path : str or Path
            Path to configuration file (JSON or YAML)
            
        Returns
        -------
        DIContainerConfig
            Loaded configuration
            
        Raises
        ------
        ConfigurationError
            If file cannot be loaded or parsed
        """
        path = Path(config_path)
        
        if not path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {config_path}",
                suggestion="Check the file path and ensure the file exists",
            )
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                if path.suffix.lower() == '.json':
                    data = json.load(f)
                elif path.suffix.lower() in ['.yml', '.yaml']:
                    data = yaml.safe_load(f)
                else:
                    raise ConfigurationError(
                        f"Unsupported configuration file format: {path.suffix}",
                        suggestion="Use .json, .yml, or .yaml files",
                    )
                    
            return cls.from_dict(data)
            
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise ConfigurationError(
                f"Failed to parse configuration file '{config_path}': {e}",
                suggestion="Check file syntax and format",
            ) from e
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration from '{config_path}': {e}",
                suggestion="Check file permissions and content",
            ) from e
            
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DIContainerConfig":
        """
        Create configuration from dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Configuration data
            
        Returns
        -------
        DIContainerConfig
            Configuration instance
        """
        # Extract component configurations
        security_data = data.pop('security', {})
        validation_data = data.pop('validation', {})
        caching_data = data.pop('caching', {})
        monitoring_data = data.pop('monitoring', {})
        resilience_data = data.pop('resilience', {})
        performance_data = data.pop('performance', {})
        logging_data = data.pop('logging', {})
        
        return cls(
            container_name=data.get('container_name', 'default'),
            version=data.get('version', '1.0.0'),
            description=data.get('description'),
            security=SecurityConfig(**security_data),
            validation=ValidationConfig(**validation_data),
            caching=CacheConfig(**caching_data),
            monitoring=MonitoringConfig(**monitoring_data),
            resilience=ResilienceConfig(**resilience_data),
            performance=PerformanceConfig(**performance_data),
            logging=LoggingConfig(**logging_data),
            services=data.get('services', {}),
            enable_env_overrides=data.get('enable_env_overrides', True),
            env_prefix=data.get('env_prefix', 'NAUTILUS_DI_'),
        )
        
    @classmethod
    def from_environment(cls, prefix: str = "NAUTILUS_DI_") -> "DIContainerConfig":
        """
        Create configuration from environment variables.
        
        Parameters
        ----------
        prefix : str
            Environment variable prefix
            
        Returns
        -------
        DIContainerConfig
            Configuration with environment overrides
        """
        config = cls()
        config._apply_environment_overrides(prefix)
        return config
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
        
    def to_file(self, config_path: Union[str, Path], format: str = "yaml") -> None:
        """
        Save configuration to file.
        
        Parameters
        ----------
        config_path : str or Path
            Output file path
        format : str
            Output format: 'json' or 'yaml'
        """
        path = Path(config_path)
        data = self.to_dict()
        
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            if format.lower() == 'json':
                json.dump(data, f, indent=2, default=str)
            elif format.lower() in ['yml', 'yaml']:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
    def apply_environment_overrides(self, prefix: Optional[str] = None) -> None:
        """
        Apply environment variable overrides.
        
        Parameters
        ----------
        prefix : str, optional
            Environment variable prefix (uses config default if not provided)
        """
        if not self.enable_env_overrides:
            return
            
        prefix = prefix or self.env_prefix
        self._apply_environment_overrides(prefix)
        
    def _apply_environment_overrides(self, prefix: str) -> None:
        """Apply environment variable overrides with prefix."""
        logger = Logger(self.__class__.__name__)
        
        # Map of environment variables to config paths
        env_mappings = {
            f"{prefix}SECURITY_STRICT_MODE": ("security", "strict_mode", bool),
            f"{prefix}SECURITY_AUDIT_IMPORTS": ("security", "audit_imports", bool),
            f"{prefix}VALIDATION_ENABLE_GRAPH": ("validation", "enable_graph_validation", bool),
            f"{prefix}VALIDATION_MAX_DEPENDENCIES": ("validation", "max_constructor_dependencies", int),
            f"{prefix}CACHE_ENABLED": ("caching", "enabled", bool),
            f"{prefix}CACHE_MAX_SIZE": ("caching", "default_max_size", int),
            f"{prefix}CACHE_TTL": ("caching", "default_ttl_seconds", float),
            f"{prefix}MONITORING_ENABLED": ("monitoring", "enabled", bool),
            f"{prefix}MONITORING_MEMORY_WARNING": ("monitoring", "memory_warning_mb", float),
            f"{prefix}RESILIENCE_ENABLED": ("resilience", "enabled", bool),
            f"{prefix}RESILIENCE_MAX_RETRIES": ("resilience", "max_retries", int),
            f"{prefix}LOGGING_LEVEL": ("logging", "level", str),
            f"{prefix}LOGGING_LOG_TO_FILE": ("logging", "log_to_file", bool),
        }
        
        for env_var, (component, field, type_func) in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                try:
                    # Type conversion
                    if type_func == bool:
                        converted_value = value.lower() in ('true', '1', 'yes', 'on')
                    elif type_func == str and field == "level":
                        converted_value = LogLevel(value.lower())
                    else:
                        converted_value = type_func(value)
                        
                    # Set value
                    component_obj = getattr(self, component)
                    setattr(component_obj, field, converted_value)
                    
                    logger.debug(f"Applied environment override: {env_var} = {converted_value}")
                    
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid environment variable value {env_var}={value}: {e}")
                    
    def validate(self) -> List[str]:
        """
        Validate configuration.
        
        Returns
        -------
        List[str]
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate security config
        if not self.security.trusted_prefixes:
            errors.append("Security: At least one trusted prefix must be specified")
            
        if self.security.max_import_depth < 1:
            errors.append("Security: max_import_depth must be positive")
            
        # Validate validation config
        if self.validation.max_constructor_dependencies < 1:
            errors.append("Validation: max_constructor_dependencies must be positive")
            
        if self.validation.max_dependency_depth < 1:
            errors.append("Validation: max_dependency_depth must be positive")
            
        # Validate cache config
        if self.caching.default_max_size < 1:
            errors.append("Cache: default_max_size must be positive")
            
        if self.caching.default_ttl_seconds is not None and self.caching.default_ttl_seconds <= 0:
            errors.append("Cache: default_ttl_seconds must be positive")
            
        # Validate monitoring config
        if self.monitoring.memory_warning_mb >= self.monitoring.memory_critical_mb:
            errors.append("Monitoring: memory_warning_mb must be less than memory_critical_mb")
            
        # Validate resilience config
        if self.resilience.max_retries < 0:
            errors.append("Resilience: max_retries must be non-negative")
            
        if self.resilience.retry_delay <= 0:
            errors.append("Resilience: retry_delay must be positive")
            
        return errors
        
    def merge(self, other: "DIContainerConfig") -> "DIContainerConfig":
        """
        Merge with another configuration.
        
        Parameters
        ----------
        other : DIContainerConfig
            Configuration to merge
            
        Returns
        -------
        DIContainerConfig
            Merged configuration
        """
        # This is a simplified merge - could be made more sophisticated
        merged_dict = self.to_dict()
        other_dict = other.to_dict()
        
        def deep_merge(dict1: Dict, dict2: Dict) -> Dict:
            result = dict1.copy()
            for key, value in dict2.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
            
        merged_dict = deep_merge(merged_dict, other_dict)
        return DIContainerConfig.from_dict(merged_dict)


class ConfigurationManager:
    """
    Manages configuration loading, validation, and environment overrides.
    """
    
    def __init__(self) -> None:
        """Initialize configuration manager."""
        self._logger = Logger(self.__class__.__name__)
        self._configs: Dict[str, DIContainerConfig] = {}
        self._default_config: Optional[DIContainerConfig] = None
        
    def load_config(
        self,
        name: str,
        config_path: Optional[Union[str, Path]] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        apply_env_overrides: bool = True,
    ) -> DIContainerConfig:
        """
        Load and register configuration.
        
        Parameters
        ----------
        name : str
            Configuration name
        config_path : str or Path, optional
            Path to configuration file
        config_dict : Dict[str, Any], optional
            Configuration dictionary
        apply_env_overrides : bool
            Whether to apply environment overrides
            
        Returns
        -------
        DIContainerConfig
            Loaded configuration
        """
        if config_path:
            config = DIContainerConfig.from_file(config_path)
        elif config_dict:
            config = DIContainerConfig.from_dict(config_dict)
        else:
            config = DIContainerConfig()
            
        if apply_env_overrides:
            config.apply_environment_overrides()
            
        # Validate configuration
        errors = config.validate()
        if errors:
            raise ConfigurationError(
                f"Configuration validation failed: {'; '.join(errors)}",
                suggestion="Review and fix configuration errors",
            )
            
        self._configs[name] = config
        
        if not self._default_config:
            self._default_config = config
            
        self._logger.info(f"Loaded configuration '{name}'")
        return config
        
    def get_config(self, name: str = "default") -> Optional[DIContainerConfig]:
        """Get configuration by name."""
        return self._configs.get(name) or self._default_config
        
    def set_default_config(self, config: DIContainerConfig) -> None:
        """Set default configuration."""
        self._default_config = config
        
    def list_configs(self) -> List[str]:
        """List all registered configuration names."""
        return list(self._configs.keys())
        
    def create_profile(self, name: str, base_config: str = "default", **overrides) -> DIContainerConfig:
        """
        Create configuration profile with overrides.
        
        Parameters
        ----------
        name : str
            Profile name
        base_config : str
            Base configuration name
        **overrides
            Configuration overrides
            
        Returns
        -------
        DIContainerConfig
            Created profile configuration
        """
        base = self.get_config(base_config)
        if not base:
            raise ConfigurationError(f"Base configuration '{base_config}' not found")
            
        # Apply overrides
        profile_dict = base.to_dict()
        profile_dict.update(overrides)
        
        profile = DIContainerConfig.from_dict(profile_dict)
        self._configs[name] = profile
        
        self._logger.info(f"Created configuration profile '{name}' based on '{base_config}'")
        return profile


# Global configuration manager
_config_manager: Optional[ConfigurationManager] = None


def get_config_manager() -> ConfigurationManager:
    """Get global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager


def load_config(
    name: str = "default",
    config_path: Optional[Union[str, Path]] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    apply_env_overrides: bool = True,
) -> DIContainerConfig:
    """Load configuration using global manager."""
    return get_config_manager().load_config(name, config_path, config_dict, apply_env_overrides)


def get_config(name: str = "default") -> Optional[DIContainerConfig]:
    """Get configuration using global manager."""
    return get_config_manager().get_config(name)