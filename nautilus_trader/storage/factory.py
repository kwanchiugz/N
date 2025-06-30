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
Storage factory for creating storage providers.
"""

from typing import Dict, Type, Optional
from enum import Enum

from nautilus_trader.storage.base import StorageProvider, StorageConfig
from nautilus_trader.storage.providers import (
    FileStorageProvider,
    FileStorageConfig,
    DatabaseStorageProvider,
    DatabaseStorageConfig,
    RedisStorageProvider,
    RedisStorageConfig,
    S3StorageProvider,
    S3StorageConfig,
)


class StorageType(str, Enum):
    """Supported storage types."""
    FILE = "file"
    DATABASE = "database"
    REDIS = "redis"
    S3 = "s3"
    MEMORY = "memory"  # For testing


class StorageFactory:
    """
    Factory for creating storage providers.
    
    This factory provides a centralized way to create and configure
    different storage providers based on configuration.
    """
    
    _providers: Dict[StorageType, Type[StorageProvider]] = {
        StorageType.FILE: FileStorageProvider,
        StorageType.DATABASE: DatabaseStorageProvider,
        StorageType.REDIS: RedisStorageProvider,
        StorageType.S3: S3StorageProvider,
    }
    
    _configs: Dict[StorageType, Type[StorageConfig]] = {
        StorageType.FILE: FileStorageConfig,
        StorageType.DATABASE: DatabaseStorageConfig,
        StorageType.REDIS: RedisStorageConfig,
        StorageType.S3: S3StorageConfig,
    }
    
    @classmethod
    def create(
        cls,
        storage_type: StorageType,
        config: Optional[StorageConfig] = None,
        **kwargs,
    ) -> StorageProvider:
        """
        Create a storage provider instance.
        
        Parameters
        ----------
        storage_type : StorageType
            The type of storage provider to create
        config : StorageConfig, optional
            Pre-configured storage config
        **kwargs
            Configuration parameters to pass to the config class
            
        Returns
        -------
        StorageProvider
            The configured storage provider instance
            
        Raises
        ------
        ValueError
            If storage type is not supported
        """
        if storage_type not in cls._providers:
            raise ValueError(f"Unsupported storage type: {storage_type}")
            
        provider_class = cls._providers[storage_type]
        
        # Create config if not provided
        if config is None:
            config_class = cls._configs[storage_type]
            config = config_class(**kwargs)
            
        return provider_class(config)
        
    @classmethod
    def register_provider(
        cls,
        storage_type: StorageType,
        provider_class: Type[StorageProvider],
        config_class: Type[StorageConfig],
    ) -> None:
        """
        Register a custom storage provider.
        
        Parameters
        ----------
        storage_type : StorageType
            The storage type identifier
        provider_class : Type[StorageProvider]
            The provider class
        config_class : Type[StorageConfig]
            The configuration class
        """
        cls._providers[storage_type] = provider_class
        cls._configs[storage_type] = config_class
        
    @classmethod
    def create_from_config(cls, config_dict: Dict[str, Any]) -> StorageProvider:
        """
        Create storage provider from configuration dictionary.
        
        Parameters
        ----------
        config_dict : Dict[str, Any]
            Configuration dictionary with 'type' and provider-specific settings
            
        Returns
        -------
        StorageProvider
            The configured storage provider instance
        """
        storage_type = StorageType(config_dict.pop("type"))
        return cls.create(storage_type, **config_dict)


# Convenience functions
async def create_storage(
    storage_type: StorageType = StorageType.FILE,
    auto_connect: bool = True,
    **kwargs,
) -> StorageProvider:
    """
    Create and optionally connect a storage provider.
    
    Parameters
    ----------
    storage_type : StorageType
        The type of storage provider
    auto_connect : bool
        Whether to automatically connect
    **kwargs
        Configuration parameters
        
    Returns
    -------
    StorageProvider
        The connected storage provider
    """
    provider = StorageFactory.create(storage_type, **kwargs)
    
    if auto_connect:
        await provider.connect()
        
    return provider


# Memory storage for testing
class MemoryStorageProvider(StorageProvider):
    """
    In-memory storage provider for testing.
    
    This provider stores everything in memory and is useful
    for unit tests and development.
    """
    
    def __init__(self, config: StorageConfig) -> None:
        """Initialize memory storage."""
        super().__init__(config)
        self._data: Dict[str, Any] = {}
        
    async def connect(self) -> None:
        """No connection needed for memory storage."""
        self._logger.info("Connected to memory storage")
        
    async def disconnect(self) -> None:
        """Clear memory on disconnect."""
        self._data.clear()
        self._logger.info("Disconnected from memory storage")
        
    async def save(self, key: str, data: Any, namespace: Optional[str] = None) -> None:
        """Save to memory."""
        storage_key = self._build_cache_key(key, namespace)
        self._data[storage_key] = data
        
    async def load(self, key: str, namespace: Optional[str] = None) -> Any:
        """Load from memory."""
        storage_key = self._build_cache_key(key, namespace)
        
        if storage_key not in self._data:
            raise StorageNotFoundError(f"Key not found: {key}")
            
        return self._data[storage_key]
        
    async def exists(self, key: str, namespace: Optional[str] = None) -> bool:
        """Check if exists in memory."""
        storage_key = self._build_cache_key(key, namespace)
        return storage_key in self._data
        
    async def delete(self, key: str, namespace: Optional[str] = None) -> bool:
        """Delete from memory."""
        storage_key = self._build_cache_key(key, namespace)
        
        if storage_key in self._data:
            del self._data[storage_key]
            return True
        return False
        
    async def list_keys(self, namespace: Optional[str] = None, pattern: Optional[str] = None) -> List[str]:
        """List keys in memory."""
        keys = []
        
        for storage_key in self._data.keys():
            if namespace and not storage_key.startswith(f"{namespace}:"):
                continue
                
            # Extract original key
            if namespace:
                key = storage_key.replace(f"{namespace}:", "", 1)
            else:
                key = storage_key
                
            if pattern:
                import fnmatch
                if fnmatch.fnmatch(key, pattern):
                    keys.append(key)
            else:
                keys.append(key)
                
        return keys
        
    async def clear(self, namespace: Optional[str] = None) -> int:
        """Clear memory storage."""
        if namespace:
            keys_to_remove = [k for k in self._data.keys() if k.startswith(f"{namespace}:")]
            for k in keys_to_remove:
                del self._data[k]
            return len(keys_to_remove)
        else:
            count = len(self._data)
            self._data.clear()
            return count


# Register memory provider
StorageFactory.register_provider(
    StorageType.MEMORY,
    MemoryStorageProvider,
    StorageConfig,
)