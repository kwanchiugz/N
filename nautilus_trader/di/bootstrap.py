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
Bootstrap configuration for dependency injection.
"""

from typing import Optional, Dict, Any
import os
import json
import yaml
from pathlib import Path

from nautilus_trader.common.component import Logger
from nautilus_trader.di.container import DIContainer, ContainerBuilder, get_container
from nautilus_trader.di.registry import ServiceRegistry
from nautilus_trader.di.exceptions import ConfigurationError
from nautilus_trader.di.module_validator import ModuleValidator
from nautilus_trader.di.graph_validator import ServiceGraphValidator

# Dynamic service catalog import
from nautilus_trader.di.service_catalog import (
    ServiceCatalog,
    ServiceDefinition,
    ServiceCategory,
    get_service_catalog,
)


class Bootstrap:
    """
    Bootstrap configuration for the DI container.
    
    Provides initialization and configuration of core services.
    """
    
    def __init__(
        self, 
        container: Optional[DIContainer] = None,
        module_validator: Optional[ModuleValidator] = None,
        service_catalog: Optional[ServiceCatalog] = None,
    ) -> None:
        """
        Initialize bootstrap.
        
        Parameters
        ----------
        container : DIContainer, optional
            Container to configure (uses global if not provided)
        module_validator : ModuleValidator, optional
            Module validator for secure imports
        service_catalog : ServiceCatalog, optional
            Service catalog for dynamic configuration
        """
        self._container = container or get_container()
        self._logger = Logger(self.__class__.__name__)
        self._module_validator = module_validator or ModuleValidator()
        self._registry = ServiceRegistry(self._container, self._module_validator)
        self._service_catalog = service_catalog or get_service_catalog()
        
    def configure_core_services(self) -> "Bootstrap":
        """Configure core Nautilus services using service catalog."""
        return self.configure_services_by_category(ServiceCategory.CORE)
        
    def configure_services_by_category(self, category: ServiceCategory) -> "Bootstrap":
        """
        Configure all services in a category using the service catalog.
        
        Parameters
        ----------
        category : ServiceCategory
            Category of services to configure
        """
        services = self._service_catalog.get_services_by_category(category)
        order = self._get_category_dependency_order(services)
        
        for service_name in order:
            service_def = self._service_catalog.get_service(service_name)
            if service_def and service_def.enabled:
                self._configure_service(service_def)
                
        self._logger.info(f"Configured {len(order)} services in category {category.value}")
        return self
        
    def configure_service_by_name(self, service_name: str, **config_overrides) -> "Bootstrap":
        """
        Configure a specific service by name.
        
        Parameters
        ----------
        service_name : str
            Name of service to configure
        **config_overrides
            Configuration overrides
        """
        service_def = self._service_catalog.get_service(service_name)
        if not service_def:
            raise ConfigurationError(
                f"Service '{service_name}' not found in catalog",
                suggestion="Check service name or register the service first",
            )
            
        if not service_def.enabled:
            self._logger.warning(f"Service '{service_name}' is disabled, skipping configuration")
            return self
            
        # Apply configuration overrides
        if config_overrides:
            service_def.configuration.update(config_overrides)
            
        self._configure_service(service_def)
        self._logger.info(f"Configured service: {service_name}")
        return self
        
    def configure_auth_services(
        self,
        secret_key: Optional[str] = None,
    ) -> "Bootstrap":
        """Configure authentication services."""
        # Configure auth services with overrides
        config_overrides = {}
        if secret_key:
            config_overrides["secret_key"] = secret_key
            
        return self.configure_services_by_category(ServiceCategory.AUTH)
        
    def configure_ai_services(
        self,
        provider: str = "deepseek",
        **config,
    ) -> "Bootstrap":
        """Configure AI services."""
        service_name = f"{provider}_provider"
        return self.configure_service_by_name(service_name, **config)
        
    def configure_from_file(self, config_path: str) -> "Bootstrap":
        """
        Configure from configuration file.
        
        Parameters
        ----------
        config_path : str
            Path to configuration file (JSON or YAML)
        """
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        # Load configuration
        with open(path) as f:
            if path.suffix == ".json":
                config = json.load(f)
            elif path.suffix in [".yml", ".yaml"]:
                config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")
                
        # Apply configuration
        self._apply_config(config)
        
        self._logger.info(f"Configured from file: {config_path}")
        return self
        
    def configure_from_environment(self) -> "Bootstrap":
        """Configure from environment variables."""
        # Storage configuration
        storage_type = os.environ.get("STORAGE_TYPE", "file")
        
        if storage_type == "redis":
            self._container.register_singleton(
                StorageProvider,
                factory=lambda: StorageFactory.create(
                    StorageType.REDIS,
                    host=os.environ.get("REDIS_HOST", "localhost"),
                    port=int(os.environ.get("REDIS_PORT", "6379")),
                    password=os.environ.get("REDIS_PASSWORD"),
                ),
            )
        elif storage_type == "database":
            self._container.register_singleton(
                StorageProvider,
                factory=lambda: StorageFactory.create(
                    StorageType.DATABASE,
                    connection_string=os.environ.get("DATABASE_URL"),
                ),
            )
            
        # AI configuration
        if os.environ.get("DEEPSEEK_API_KEY"):
            self.configure_ai_services(
                provider="deepseek",
                api_key=os.environ.get("DEEPSEEK_API_KEY"),
            )
            
        self._logger.info("Configured from environment")
        return self
        
    def configure_module_security(
        self,
        trusted_prefixes: Optional[List[str]] = None,
        strict_mode: bool = True,
        audit_imports: bool = True,
    ) -> "Bootstrap":
        """
        Configure module security settings.
        
        Parameters
        ----------
        trusted_prefixes : List[str], optional
            List of trusted module path prefixes
        strict_mode : bool
            If True, enforce strict validation (raise exceptions on failures)
        audit_imports : bool
            If True, log all import attempts for security auditing
        """
        if trusted_prefixes:
            self._module_validator._trusted_prefixes = trusted_prefixes
            
        self._module_validator.set_strict_mode(strict_mode)
        self._module_validator._audit_imports = audit_imports
        
        self._logger.info(
            f"Configured module security: strict_mode={strict_mode}, "
            f"audit_imports={audit_imports}, trusted_prefixes={trusted_prefixes}"
        )
        return self
        
    def auto_discover(self, *module_names: str) -> "Bootstrap":
        """
        Auto-discover and register services from modules.
        
        Parameters
        ----------
        *module_names : str
            Modules to scan for services
        """
        total = 0
        
        for module_name in module_names:
            count = self._registry.scan_module(module_name)
            total += count
            
        self._logger.info(f"Auto-discovered {total} services")
        return self
        
    def validate(self, validate_graph: bool = True, partial_validation: bool = False) -> bool:
        """
        Validate container configuration.
        
        Parameters
        ----------
        validate_graph : bool
            If True, perform comprehensive dependency graph validation
        partial_validation : bool
            If True, allow some services to be unresolvable (useful during development)
        
        Returns
        -------
        bool
            True if valid
            
        Raises
        ------
        ConfigurationError
            If validation fails
        """
        # Check core services
        required_services = [
            EventBus,
            StorageProvider,
        ]
        
        missing_services = []
        for service in required_services:
            try:
                self._container.resolve(service)
            except ValueError:
                missing_services.append(service.__name__)
                
        if missing_services:
            raise ConfigurationError(
                f"Missing required services: {', '.join(missing_services)}",
                suggestion="Ensure all required services are registered before validation",
            )
        
        # Perform dependency graph validation if requested
        if validate_graph:
            self._logger.info("Performing dependency graph validation")
            validator = ServiceGraphValidator()
            result = validator.validate(self._container, partial=partial_validation)
            
            if not result.is_valid:
                error_details = result.format_report()
                raise ConfigurationError(
                    f"Service graph validation failed:\n{error_details}",
                    suggestion="Fix dependency issues before proceeding",
                )
            
            if result.has_warnings():
                self._logger.warning(f"Graph validation completed with warnings:\n{result.format_report()}")
            else:
                self._logger.info(f"Graph validation passed: {result.services_validated} services validated")
                
        self._logger.info("Container configuration validated")
        return True
        
    def _configure_service(self, service_def: ServiceDefinition) -> None:
        """
        Configure a single service using its definition.
        
        Parameters
        ----------
        service_def : ServiceDefinition
            Service definition to configure
        """
        try:
            # Validate service definition
            self._service_catalog.validate_service(service_def)
            
            # Resolve interface
            interface = self._service_catalog.resolve_interface(service_def.interface_path)
            
            # Resolve implementation or factory
            implementation = None
            factory = None
            
            if service_def.implementation_path:
                implementation = self._service_catalog.resolve_implementation(service_def.implementation_path)
                
            if service_def.factory_path:
                factory = self._service_catalog.resolve_factory(service_def.factory_path)
                
            # Apply configuration if available
            if service_def.configuration and implementation:
                # Create factory that applies configuration
                original_factory = factory
                def configured_factory():
                    if original_factory:
                        instance = original_factory()
                    else:
                        instance = self._container.create_instance(implementation)
                    
                    # Apply configuration
                    for key, value in service_def.configuration.items():
                        if hasattr(instance, key):
                            setattr(instance, key, value)
                    
                    return instance
                factory = configured_factory
                implementation = None  # Use factory instead
                
            # Register in container
            self._container.register(
                interface=interface,
                implementation=implementation,
                factory=factory,
                lifetime=service_def.lifetime,
            )
            
            self._logger.debug(f"Configured service '{service_def.name}' with lifetime {service_def.lifetime.value}")
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to configure service '{service_def.name}': {e}",
                service_class=service_def.name,
                suggestion="Check service definition and dependencies",
            ) from e
            
    def _get_category_dependency_order(self, services: List[ServiceDefinition]) -> List[str]:
        """
        Get services in dependency order for a category.
        
        Parameters
        ----------
        services : List[ServiceDefinition]
            Services to order
            
        Returns
        -------
        List[str]
            Service names in dependency order
        """
        # Build dependency graph for this category
        service_names = {svc.name for svc in services}
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(service_name: str):
            if service_name in temp_visited:
                raise ConfigurationError(
                    f"Circular dependency detected in category services involving '{service_name}'",
                    suggestion="Review service dependencies within the category",
                )
            if service_name in visited:
                return
                
            temp_visited.add(service_name)
            
            # Find service definition
            service_def = next((svc for svc in services if svc.name == service_name), None)
            if service_def:
                # Visit dependencies that are in this category
                for dep in service_def.dependencies:
                    if dep in service_names:
                        visit(dep)
                        
            temp_visited.remove(service_name)
            visited.add(service_name)
            order.append(service_name)
            
        # Visit all services in category
        for service in services:
            if service.name not in visited:
                visit(service.name)
                
        return order
        
    def _apply_config(self, config: Dict[str, Any]) -> None:
        """Apply configuration dictionary."""
        # Services configuration
        if "services" in config:
            self._registry.register_assembly({"services": config["services"]})
            
        # Storage configuration
        if "storage" in config:
            storage_config = config["storage"]
            storage_type = StorageType(storage_config.pop("type", "file"))
            
            self._container.register_singleton(
                StorageProvider,
                factory=lambda: StorageFactory.create(storage_type, **storage_config),
            )
            
        # AI configuration
        if "ai" in config:
            ai_config = config["ai"]
            provider = ai_config.pop("provider", "deepseek")
            self.configure_ai_services(provider, **ai_config)


# Convenience function
def bootstrap_application(
    config_path: Optional[str] = None,
    use_environment: bool = True,
    auto_discover_modules: Optional[list] = None,
) -> DIContainer:
    """
    Bootstrap the application with standard configuration.
    
    Parameters
    ----------
    config_path : str, optional
        Configuration file path
    use_environment : bool
        Whether to use environment variables
    auto_discover_modules : list, optional
        Modules to auto-discover
        
    Returns
    -------
    DIContainer
        Configured container
    """
    bootstrap = Bootstrap()
    
    # Configure core services
    bootstrap.configure_core_services()
    
    # Load from config file
    if config_path:
        bootstrap.configure_from_file(config_path)
        
    # Load from environment
    if use_environment:
        bootstrap.configure_from_environment()
        
    # Auto-discover
    if auto_discover_modules:
        bootstrap.auto_discover(*auto_discover_modules)
        
    # Validate - this will raise ConfigurationError if validation fails
    bootstrap.validate()
        
    return bootstrap._container