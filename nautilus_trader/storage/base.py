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
Base storage provider interface and exceptions.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar
from datetime import datetime
import json
import pickle

from nautilus_trader.config import NautilusConfig
from nautilus_trader.common.component import Logger
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.cryptography.encryption import DataEncryptor


T = TypeVar("T")


class StorageError(Exception):
    """Base exception for storage-related errors."""
    pass


class StorageNotFoundError(StorageError):
    """Raised when requested data is not found in storage."""
    pass


class StorageConfig(NautilusConfig):
    """Base configuration for storage providers."""
    
    encryption_enabled: bool = False
    encryption_key: Optional[str] = None
    compression_enabled: bool = True
    cache_enabled: bool = True
    cache_ttl: int = 300  # seconds
    retry_attempts: int = 3
    retry_delay: float = 1.0  # seconds


class StorageProvider(ABC):
    """
    Abstract base class for storage providers.
    
    This interface provides a unified API for different storage backends
    including file systems, databases, cloud storage, and cache systems.
    """
    
    def __init__(self, config: StorageConfig) -> None:
        """Initialize the storage provider."""
        self._config = config
        self._logger = Logger(self.__class__.__name__)
        self._encryptor = None
        
        if config.encryption_enabled:
            if not config.encryption_key:
                raise ValueError("Encryption key required when encryption is enabled")
            self._encryptor = DataEncryptor(config.encryption_key.encode())
            
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, int] = {}
        
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the storage backend."""
        pass
        
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the storage backend."""
        pass
        
    @abstractmethod
    async def save(self, key: str, data: Any, namespace: Optional[str] = None) -> None:
        """
        Save data to storage.
        
        Parameters
        ----------
        key : str
            Unique identifier for the data
        data : Any
            Data to store (must be serializable)
        namespace : str, optional
            Optional namespace for data organization
        """
        pass
        
    @abstractmethod
    async def load(self, key: str, namespace: Optional[str] = None) -> Any:
        """
        Load data from storage.
        
        Parameters
        ----------
        key : str
            Unique identifier for the data
        namespace : str, optional
            Optional namespace for data organization
            
        Returns
        -------
        Any
            The stored data
            
        Raises
        ------
        StorageNotFoundError
            If the key does not exist
        """
        pass
        
    @abstractmethod
    async def exists(self, key: str, namespace: Optional[str] = None) -> bool:
        """
        Check if a key exists in storage.
        
        Parameters
        ----------
        key : str
            Unique identifier to check
        namespace : str, optional
            Optional namespace
            
        Returns
        -------
        bool
            True if the key exists
        """
        pass
        
    @abstractmethod
    async def delete(self, key: str, namespace: Optional[str] = None) -> bool:
        """
        Delete data from storage.
        
        Parameters
        ----------
        key : str
            Unique identifier for the data
        namespace : str, optional
            Optional namespace
            
        Returns
        -------
        bool
            True if deletion was successful
        """
        pass
        
    @abstractmethod
    async def list_keys(self, namespace: Optional[str] = None, pattern: Optional[str] = None) -> List[str]:
        """
        List all keys in storage.
        
        Parameters
        ----------
        namespace : str, optional
            Filter by namespace
        pattern : str, optional
            Filter keys by pattern (e.g., "user_*")
            
        Returns
        -------
        List[str]
            List of matching keys
        """
        pass
        
    @abstractmethod
    async def clear(self, namespace: Optional[str] = None) -> int:
        """
        Clear all data from storage.
        
        Parameters
        ----------
        namespace : str, optional
            Clear only specified namespace
            
        Returns
        -------
        int
            Number of items cleared
        """
        pass
        
    # Typed convenience methods
    async def save_json(self, key: str, data: Dict[str, Any], namespace: Optional[str] = None) -> None:
        """Save JSON-serializable data."""
        json_data = json.dumps(data)
        await self.save(key, json_data, namespace)
        
    async def load_json(self, key: str, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Load and parse JSON data."""
        json_data = await self.load(key, namespace)
        return json.loads(json_data)
        
    async def save_object(self, key: str, obj: Any, namespace: Optional[str] = None) -> None:
        """Save Python object using pickle."""
        pickle_data = pickle.dumps(obj)
        await self.save(key, pickle_data, namespace)
        
    async def load_object(self, key: str, obj_type: Type[T], namespace: Optional[str] = None) -> T:
        """Load and unpickle Python object."""
        pickle_data = await self.load(key, namespace)
        obj = pickle.loads(pickle_data)
        if not isinstance(obj, obj_type):
            raise TypeError(f"Expected {obj_type}, got {type(obj)}")
        return obj
        
    # Batch operations
    async def save_batch(self, items: Dict[str, Any], namespace: Optional[str] = None) -> None:
        """
        Save multiple items in batch.
        
        Parameters
        ----------
        items : Dict[str, Any]
            Dictionary of key-value pairs to save
        namespace : str, optional
            Optional namespace
        """
        for key, data in items.items():
            await self.save(key, data, namespace)
            
    async def load_batch(self, keys: List[str], namespace: Optional[str] = None) -> Dict[str, Any]:
        """
        Load multiple items in batch.
        
        Parameters
        ----------
        keys : List[str]
            List of keys to load
        namespace : str, optional
            Optional namespace
            
        Returns
        -------
        Dict[str, Any]
            Dictionary of key-value pairs
        """
        result = {}
        for key in keys:
            try:
                result[key] = await self.load(key, namespace)
            except StorageNotFoundError:
                continue
        return result
        
    # Encryption helpers
    def _encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data if encryption is enabled."""
        if self._encryptor:
            return self._encryptor.encrypt(data)
        return data
        
    def _decrypt_data(self, data: bytes) -> bytes:
        """Decrypt data if encryption is enabled."""
        if self._encryptor:
            return self._encryptor.decrypt(data)
        return data
        
    # Cache helpers
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get data from cache if enabled and not expired."""
        if not self._config.cache_enabled:
            return None
            
        if cache_key not in self._cache:
            return None
            
        timestamp = self._cache_timestamps.get(cache_key, 0)
        now = dt_to_unix_nanos(datetime.utcnow())
        age_seconds = (now - timestamp) / 1e9
        
        if age_seconds > self._config.cache_ttl:
            del self._cache[cache_key]
            del self._cache_timestamps[cache_key]
            return None
            
        return self._cache[cache_key]
        
    def _put_in_cache(self, cache_key: str, data: Any) -> None:
        """Put data in cache if enabled."""
        if self._config.cache_enabled:
            self._cache[cache_key] = data
            self._cache_timestamps[cache_key] = dt_to_unix_nanos(datetime.utcnow())
            
    def _build_cache_key(self, key: str, namespace: Optional[str] = None) -> str:
        """Build cache key from key and namespace."""
        if namespace:
            return f"{namespace}:{key}"
        return key