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
Storage provider implementations.
"""

import os
import json
import asyncio
import aiofiles
from pathlib import Path
from typing import Any, Dict, List, Optional
import aioboto3
import aioredis
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

from nautilus_trader.storage.base import (
    StorageProvider,
    StorageConfig,
    StorageError,
    StorageNotFoundError,
)


class FileStorageConfig(StorageConfig):
    """Configuration for file-based storage."""
    
    base_path: str = "./data"
    file_extension: str = ".json"
    use_subdirectories: bool = True
    max_file_size_mb: int = 100


class FileStorageProvider(StorageProvider):
    """
    File system based storage provider.
    
    Stores data as files on the local filesystem with optional
    directory structure for namespaces.
    """
    
    def __init__(self, config: FileStorageConfig) -> None:
        """Initialize file storage provider."""
        super().__init__(config)
        self._config: FileStorageConfig = config
        self._base_path = Path(config.base_path)
        
    async def connect(self) -> None:
        """Create base directory if it doesn't exist."""
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._logger.info(f"Connected to file storage at {self._base_path}")
        
    async def disconnect(self) -> None:
        """No cleanup needed for file storage."""
        self._logger.info("Disconnected from file storage")
        
    async def save(self, key: str, data: Any, namespace: Optional[str] = None) -> None:
        """Save data to file."""
        file_path = self._get_file_path(key, namespace)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Serialize data
        if isinstance(data, bytes):
            content = data
        elif isinstance(data, str):
            content = data.encode()
        else:
            content = json.dumps(data).encode()
            
        # Encrypt if enabled
        content = self._encrypt_data(content)
        
        # Write to file
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)
            
        self._logger.debug(f"Saved data to {file_path}")
        
    async def load(self, key: str, namespace: Optional[str] = None) -> Any:
        """Load data from file."""
        # Check cache first
        cache_key = self._build_cache_key(key, namespace)
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
            
        file_path = self._get_file_path(key, namespace)
        
        if not file_path.exists():
            raise StorageNotFoundError(f"Key not found: {key}")
            
        # Read from file
        async with aiofiles.open(file_path, 'rb') as f:
            content = await f.read()
            
        # Decrypt if enabled
        content = self._decrypt_data(content)
        
        # Deserialize
        try:
            data = json.loads(content.decode())
        except json.JSONDecodeError:
            data = content
            
        # Cache the result
        self._put_in_cache(cache_key, data)
        
        return data
        
    async def exists(self, key: str, namespace: Optional[str] = None) -> bool:
        """Check if file exists."""
        file_path = self._get_file_path(key, namespace)
        return file_path.exists()
        
    async def delete(self, key: str, namespace: Optional[str] = None) -> bool:
        """Delete file."""
        file_path = self._get_file_path(key, namespace)
        
        if file_path.exists():
            file_path.unlink()
            
            # Clear from cache
            cache_key = self._build_cache_key(key, namespace)
            self._cache.pop(cache_key, None)
            self._cache_timestamps.pop(cache_key, None)
            
            return True
        return False
        
    async def list_keys(self, namespace: Optional[str] = None, pattern: Optional[str] = None) -> List[str]:
        """List files in directory."""
        if namespace and self._config.use_subdirectories:
            search_path = self._base_path / namespace
        else:
            search_path = self._base_path
            
        if not search_path.exists():
            return []
            
        keys = []
        for file_path in search_path.rglob(f"*{self._config.file_extension}"):
            if file_path.is_file():
                key = file_path.stem
                if pattern:
                    import fnmatch
                    if fnmatch.fnmatch(key, pattern):
                        keys.append(key)
                else:
                    keys.append(key)
                    
        return keys
        
    async def clear(self, namespace: Optional[str] = None) -> int:
        """Clear files from directory."""
        if namespace and self._config.use_subdirectories:
            clear_path = self._base_path / namespace
        else:
            clear_path = self._base_path
            
        if not clear_path.exists():
            return 0
            
        count = 0
        for file_path in clear_path.rglob(f"*{self._config.file_extension}"):
            if file_path.is_file():
                file_path.unlink()
                count += 1
                
        # Clear cache
        if namespace:
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(f"{namespace}:")]
            for k in keys_to_remove:
                self._cache.pop(k, None)
                self._cache_timestamps.pop(k, None)
        else:
            self._cache.clear()
            self._cache_timestamps.clear()
            
        return count
        
    def _get_file_path(self, key: str, namespace: Optional[str] = None) -> Path:
        """Build file path from key and namespace."""
        # Sanitize key for filesystem
        safe_key = key.replace("/", "_").replace("\\", "_")
        
        if namespace and self._config.use_subdirectories:
            return self._base_path / namespace / f"{safe_key}{self._config.file_extension}"
        else:
            if namespace:
                return self._base_path / f"{namespace}_{safe_key}{self._config.file_extension}"
            return self._base_path / f"{safe_key}{self._config.file_extension}"


class DatabaseStorageConfig(StorageConfig):
    """Configuration for database storage."""
    
    connection_string: str
    table_name: str = "nautilus_storage"
    pool_size: int = 10
    max_overflow: int = 20


class DatabaseStorageProvider(StorageProvider):
    """
    Database-based storage provider using SQLAlchemy.
    
    Supports PostgreSQL, MySQL, and SQLite.
    """
    
    def __init__(self, config: DatabaseStorageConfig) -> None:
        """Initialize database storage provider."""
        super().__init__(config)
        self._config: DatabaseStorageConfig = config
        self._engine = None
        self._session_factory = None
        
    async def connect(self) -> None:
        """Create database connection pool."""
        self._engine = create_async_engine(
            self._config.connection_string,
            pool_size=self._config.pool_size,
            max_overflow=self._config.max_overflow,
        )
        
        self._session_factory = sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        
        # Create table if not exists
        async with self._engine.begin() as conn:
            await conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {self._config.table_name} (
                    namespace VARCHAR(255),
                    key VARCHAR(255),
                    value BYTEA,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (namespace, key)
                )
            """))
            
        self._logger.info("Connected to database storage")
        
    async def disconnect(self) -> None:
        """Close database connections."""
        if self._engine:
            await self._engine.dispose()
        self._logger.info("Disconnected from database storage")
        
    async def save(self, key: str, data: Any, namespace: Optional[str] = None) -> None:
        """Save data to database."""
        namespace = namespace or "default"
        
        # Serialize data
        if isinstance(data, bytes):
            value = data
        else:
            value = json.dumps(data).encode()
            
        # Encrypt if enabled
        value = self._encrypt_data(value)
        
        async with self._session_factory() as session:
            # Upsert
            await session.execute(text(f"""
                INSERT INTO {self._config.table_name} (namespace, key, value, updated_at)
                VALUES (:namespace, :key, :value, CURRENT_TIMESTAMP)
                ON CONFLICT (namespace, key)
                DO UPDATE SET value = :value, updated_at = CURRENT_TIMESTAMP
            """), {"namespace": namespace, "key": key, "value": value})
            
            await session.commit()
            
    async def load(self, key: str, namespace: Optional[str] = None) -> Any:
        """Load data from database."""
        namespace = namespace or "default"
        
        # Check cache first
        cache_key = self._build_cache_key(key, namespace)
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
            
        async with self._session_factory() as session:
            result = await session.execute(text(f"""
                SELECT value FROM {self._config.table_name}
                WHERE namespace = :namespace AND key = :key
            """), {"namespace": namespace, "key": key})
            
            row = result.first()
            
        if not row:
            raise StorageNotFoundError(f"Key not found: {key}")
            
        # Decrypt if enabled
        value = self._decrypt_data(row[0])
        
        # Deserialize
        try:
            data = json.loads(value.decode())
        except:
            data = value
            
        # Cache the result
        self._put_in_cache(cache_key, data)
        
        return data
        
    async def exists(self, key: str, namespace: Optional[str] = None) -> bool:
        """Check if key exists in database."""
        namespace = namespace or "default"
        
        async with self._session_factory() as session:
            result = await session.execute(text(f"""
                SELECT 1 FROM {self._config.table_name}
                WHERE namespace = :namespace AND key = :key
                LIMIT 1
            """), {"namespace": namespace, "key": key})
            
            return result.first() is not None
            
    async def delete(self, key: str, namespace: Optional[str] = None) -> bool:
        """Delete from database."""
        namespace = namespace or "default"
        
        async with self._session_factory() as session:
            result = await session.execute(text(f"""
                DELETE FROM {self._config.table_name}
                WHERE namespace = :namespace AND key = :key
            """), {"namespace": namespace, "key": key})
            
            await session.commit()
            
            # Clear from cache
            cache_key = self._build_cache_key(key, namespace)
            self._cache.pop(cache_key, None)
            self._cache_timestamps.pop(cache_key, None)
            
            return result.rowcount > 0
            
    async def list_keys(self, namespace: Optional[str] = None, pattern: Optional[str] = None) -> List[str]:
        """List keys from database."""
        query = f"SELECT key FROM {self._config.table_name}"
        params = {}
        
        conditions = []
        if namespace:
            conditions.append("namespace = :namespace")
            params["namespace"] = namespace
        if pattern:
            conditions.append("key LIKE :pattern")
            params["pattern"] = pattern.replace("*", "%")
            
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
            
        async with self._session_factory() as session:
            result = await session.execute(text(query), params)
            return [row[0] for row in result]
            
    async def clear(self, namespace: Optional[str] = None) -> int:
        """Clear data from database."""
        query = f"DELETE FROM {self._config.table_name}"
        params = {}
        
        if namespace:
            query += " WHERE namespace = :namespace"
            params["namespace"] = namespace
            
        async with self._session_factory() as session:
            result = await session.execute(text(query), params)
            await session.commit()
            
            # Clear cache
            if namespace:
                keys_to_remove = [k for k in self._cache.keys() if k.startswith(f"{namespace}:")]
                for k in keys_to_remove:
                    self._cache.pop(k, None)
                    self._cache_timestamps.pop(k, None)
            else:
                self._cache.clear()
                self._cache_timestamps.clear()
                
            return result.rowcount


class RedisStorageConfig(StorageConfig):
    """Configuration for Redis storage."""
    
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    key_prefix: str = "nautilus"
    max_connections: int = 10


class RedisStorageProvider(StorageProvider):
    """
    Redis-based storage provider.
    
    Uses Redis for high-performance caching and storage with
    optional persistence.
    """
    
    def __init__(self, config: RedisStorageConfig) -> None:
        """Initialize Redis storage provider."""
        super().__init__(config)
        self._config: RedisStorageConfig = config
        self._redis = None
        
    async def connect(self) -> None:
        """Connect to Redis."""
        self._redis = await aioredis.create_redis_pool(
            f"redis://{self._config.host}:{self._config.port}",
            password=self._config.password,
            db=self._config.db,
            maxsize=self._config.max_connections,
        )
        self._logger.info("Connected to Redis storage")
        
    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._redis:
            self._redis.close()
            await self._redis.wait_closed()
        self._logger.info("Disconnected from Redis storage")
        
    async def save(self, key: str, data: Any, namespace: Optional[str] = None) -> None:
        """Save data to Redis."""
        redis_key = self._build_redis_key(key, namespace)
        
        # Serialize data
        if isinstance(data, bytes):
            value = data
        else:
            value = json.dumps(data).encode()
            
        # Encrypt if enabled
        value = self._encrypt_data(value)
        
        await self._redis.set(redis_key, value)
        
    async def load(self, key: str, namespace: Optional[str] = None) -> Any:
        """Load data from Redis."""
        redis_key = self._build_redis_key(key, namespace)
        
        value = await self._redis.get(redis_key)
        if value is None:
            raise StorageNotFoundError(f"Key not found: {key}")
            
        # Decrypt if enabled
        value = self._decrypt_data(value)
        
        # Deserialize
        try:
            return json.loads(value.decode())
        except:
            return value
            
    async def exists(self, key: str, namespace: Optional[str] = None) -> bool:
        """Check if key exists in Redis."""
        redis_key = self._build_redis_key(key, namespace)
        return await self._redis.exists(redis_key) > 0
        
    async def delete(self, key: str, namespace: Optional[str] = None) -> bool:
        """Delete from Redis."""
        redis_key = self._build_redis_key(key, namespace)
        return await self._redis.delete(redis_key) > 0
        
    async def list_keys(self, namespace: Optional[str] = None, pattern: Optional[str] = None) -> List[str]:
        """List keys from Redis."""
        search_pattern = self._build_redis_key(pattern or "*", namespace)
        
        keys = []
        cursor = b'0'
        while cursor:
            cursor, matches = await self._redis.scan(cursor, match=search_pattern)
            for key in matches:
                # Remove prefix to get original key
                original_key = key.decode().replace(f"{self._config.key_prefix}:", "", 1)
                if namespace:
                    original_key = original_key.replace(f"{namespace}:", "", 1)
                keys.append(original_key)
                
        return keys
        
    async def clear(self, namespace: Optional[str] = None) -> int:
        """Clear data from Redis."""
        pattern = self._build_redis_key("*", namespace)
        
        count = 0
        cursor = b'0'
        while cursor:
            cursor, keys = await self._redis.scan(cursor, match=pattern)
            if keys:
                count += await self._redis.delete(*keys)
                
        return count
        
    def _build_redis_key(self, key: str, namespace: Optional[str] = None) -> str:
        """Build Redis key with prefix and namespace."""
        parts = [self._config.key_prefix]
        if namespace:
            parts.append(namespace)
        parts.append(key)
        return ":".join(parts)


class S3StorageConfig(StorageConfig):
    """Configuration for S3 storage."""
    
    bucket_name: str
    region: str = "us-east-1"
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    endpoint_url: Optional[str] = None  # For S3-compatible services
    prefix: str = "nautilus"


class S3StorageProvider(StorageProvider):
    """
    AWS S3 based storage provider.
    
    Uses S3 for scalable cloud storage with high durability.
    """
    
    def __init__(self, config: S3StorageConfig) -> None:
        """Initialize S3 storage provider."""
        super().__init__(config)
        self._config: S3StorageConfig = config
        self._session = None
        self._s3 = None
        
    async def connect(self) -> None:
        """Connect to S3."""
        self._session = aioboto3.Session(
            aws_access_key_id=self._config.access_key_id,
            aws_secret_access_key=self._config.secret_access_key,
            region_name=self._config.region,
        )
        
        # Test connection
        async with self._session.client('s3', endpoint_url=self._config.endpoint_url) as s3:
            await s3.head_bucket(Bucket=self._config.bucket_name)
            
        self._logger.info(f"Connected to S3 storage (bucket: {self._config.bucket_name})")
        
    async def disconnect(self) -> None:
        """No persistent connection for S3."""
        self._logger.info("Disconnected from S3 storage")
        
    async def save(self, key: str, data: Any, namespace: Optional[str] = None) -> None:
        """Save data to S3."""
        s3_key = self._build_s3_key(key, namespace)
        
        # Serialize data
        if isinstance(data, bytes):
            body = data
        else:
            body = json.dumps(data).encode()
            
        # Encrypt if enabled
        body = self._encrypt_data(body)
        
        async with self._session.client('s3', endpoint_url=self._config.endpoint_url) as s3:
            await s3.put_object(
                Bucket=self._config.bucket_name,
                Key=s3_key,
                Body=body,
            )
            
    async def load(self, key: str, namespace: Optional[str] = None) -> Any:
        """Load data from S3."""
        # Check cache first
        cache_key = self._build_cache_key(key, namespace)
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
            
        s3_key = self._build_s3_key(key, namespace)
        
        try:
            async with self._session.client('s3', endpoint_url=self._config.endpoint_url) as s3:
                response = await s3.get_object(
                    Bucket=self._config.bucket_name,
                    Key=s3_key,
                )
                body = await response['Body'].read()
                
        except Exception as e:
            if "NoSuchKey" in str(e):
                raise StorageNotFoundError(f"Key not found: {key}")
            raise StorageError(f"Failed to load from S3: {e}")
            
        # Decrypt if enabled
        body = self._decrypt_data(body)
        
        # Deserialize
        try:
            data = json.loads(body.decode())
        except:
            data = body
            
        # Cache the result
        self._put_in_cache(cache_key, data)
        
        return data
        
    async def exists(self, key: str, namespace: Optional[str] = None) -> bool:
        """Check if object exists in S3."""
        s3_key = self._build_s3_key(key, namespace)
        
        try:
            async with self._session.client('s3', endpoint_url=self._config.endpoint_url) as s3:
                await s3.head_object(
                    Bucket=self._config.bucket_name,
                    Key=s3_key,
                )
                return True
        except:
            return False
            
    async def delete(self, key: str, namespace: Optional[str] = None) -> bool:
        """Delete object from S3."""
        s3_key = self._build_s3_key(key, namespace)
        
        try:
            async with self._session.client('s3', endpoint_url=self._config.endpoint_url) as s3:
                await s3.delete_object(
                    Bucket=self._config.bucket_name,
                    Key=s3_key,
                )
                
            # Clear from cache
            cache_key = self._build_cache_key(key, namespace)
            self._cache.pop(cache_key, None)
            self._cache_timestamps.pop(cache_key, None)
            
            return True
        except:
            return False
            
    async def list_keys(self, namespace: Optional[str] = None, pattern: Optional[str] = None) -> List[str]:
        """List objects in S3."""
        prefix = self._config.prefix
        if namespace:
            prefix = f"{prefix}/{namespace}"
            
        keys = []
        
        async with self._session.client('s3', endpoint_url=self._config.endpoint_url) as s3:
            paginator = s3.get_paginator('list_objects_v2')
            
            async for page in paginator.paginate(
                Bucket=self._config.bucket_name,
                Prefix=prefix,
            ):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        # Remove prefix to get original key
                        original_key = key.replace(f"{prefix}/", "", 1)
                        
                        if pattern:
                            import fnmatch
                            if fnmatch.fnmatch(original_key, pattern):
                                keys.append(original_key)
                        else:
                            keys.append(original_key)
                            
        return keys
        
    async def clear(self, namespace: Optional[str] = None) -> int:
        """Delete objects from S3."""
        keys = await self.list_keys(namespace)
        
        if not keys:
            return 0
            
        # S3 delete in batches (max 1000 per request)
        count = 0
        async with self._session.client('s3', endpoint_url=self._config.endpoint_url) as s3:
            for i in range(0, len(keys), 1000):
                batch = keys[i:i+1000]
                objects = [{'Key': self._build_s3_key(k, namespace)} for k in batch]
                
                response = await s3.delete_objects(
                    Bucket=self._config.bucket_name,
                    Delete={'Objects': objects},
                )
                
                count += len(response.get('Deleted', []))
                
        # Clear cache
        if namespace:
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(f"{namespace}:")]
            for k in keys_to_remove:
                self._cache.pop(k, None)
                self._cache_timestamps.pop(k, None)
        else:
            self._cache.clear()
            self._cache_timestamps.clear()
            
        return count
        
    def _build_s3_key(self, key: str, namespace: Optional[str] = None) -> str:
        """Build S3 object key with prefix and namespace."""
        parts = [self._config.prefix]
        if namespace:
            parts.append(namespace)
        parts.append(key)
        return "/".join(parts)