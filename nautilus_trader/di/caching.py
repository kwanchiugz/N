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
Advanced caching system for dependency injection container.
"""

import time
import threading
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, Set, Type, TypeVar, Union
from enum import Enum
from collections import OrderedDict

from nautilus_trader.common.component import Logger


T = TypeVar("T")


class CacheStrategy(str, Enum):
    """Cache eviction strategies."""
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    TTL = "ttl"           # Time To Live
    WEAK = "weak"         # Weak references (for memory pressure)
    NONE = "none"         # No caching


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl_seconds: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds
        
    def touch(self) -> None:
        """Update access metadata."""
        self.last_accessed = time.time()
        self.access_count += 1


class CacheEvictionPolicy(ABC):
    """Abstract base class for cache eviction policies."""
    
    @abstractmethod
    def should_evict(self, entry: CacheEntry, current_time: float) -> bool:
        """Determine if an entry should be evicted."""
        pass
        
    @abstractmethod
    def get_eviction_candidates(self, entries: Dict[Any, CacheEntry], count: int) -> Set[Any]:
        """Get candidates for eviction."""
        pass


class LRUEvictionPolicy(CacheEvictionPolicy):
    """Least Recently Used eviction policy."""
    
    def should_evict(self, entry: CacheEntry, current_time: float) -> bool:
        """LRU doesn't auto-evict based on time."""
        return False
        
    def get_eviction_candidates(self, entries: Dict[Any, CacheEntry], count: int) -> Set[Any]:
        """Get least recently used entries."""
        sorted_items = sorted(
            entries.items(),
            key=lambda x: x[1].last_accessed
        )
        return {item[0] for item in sorted_items[:count]}


class LFUEvictionPolicy(CacheEvictionPolicy):
    """Least Frequently Used eviction policy."""
    
    def should_evict(self, entry: CacheEntry, current_time: float) -> bool:
        """LFU doesn't auto-evict based on time."""
        return False
        
    def get_eviction_candidates(self, entries: Dict[Any, CacheEntry], count: int) -> Set[Any]:
        """Get least frequently used entries."""
        sorted_items = sorted(
            entries.items(),
            key=lambda x: x[1].access_count
        )
        return {item[0] for item in sorted_items[:count]}


class TTLEvictionPolicy(CacheEvictionPolicy):
    """Time To Live eviction policy."""
    
    def __init__(self, default_ttl: float = 300.0) -> None:
        """
        Initialize TTL policy.
        
        Parameters
        ----------
        default_ttl : float
            Default TTL in seconds (5 minutes)
        """
        self.default_ttl = default_ttl
        
    def should_evict(self, entry: CacheEntry, current_time: float) -> bool:
        """Check if entry has expired."""
        return entry.is_expired()
        
    def get_eviction_candidates(self, entries: Dict[Any, CacheEntry], count: int) -> Set[Any]:
        """Get expired entries first, then oldest."""
        current_time = time.time()
        
        # First get expired entries
        expired = {
            key for key, entry in entries.items()
            if entry.is_expired()
        }
        
        if len(expired) >= count:
            return set(list(expired)[:count])
            
        # If not enough expired entries, get oldest
        remaining = count - len(expired)
        sorted_items = sorted(
            entries.items(),
            key=lambda x: x[1].created_at
        )
        oldest = {item[0] for item in sorted_items[:remaining]}
        
        return expired | oldest


class ServiceCache(Generic[T]):
    """
    High-performance cache for dependency injection services.
    
    Supports multiple eviction strategies and configurable behavior.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        strategy: CacheStrategy = CacheStrategy.LRU,
        default_ttl: Optional[float] = None,
        enable_weak_refs: bool = False,
    ) -> None:
        """
        Initialize service cache.
        
        Parameters
        ----------
        max_size : int
            Maximum cache size
        strategy : CacheStrategy
            Eviction strategy
        default_ttl : float, optional
            Default TTL in seconds
        enable_weak_refs : bool
            Use weak references for cached objects
        """
        self._max_size = max_size
        self._strategy = strategy
        self._default_ttl = default_ttl
        self._enable_weak_refs = enable_weak_refs
        
        self._cache: Dict[Any, CacheEntry] = {}
        self._weak_cache: Dict[Any, weakref.ref] = {} if enable_weak_refs else None
        self._lock = threading.RLock()
        self._logger = Logger(self.__class__.__name__)
        
        # Initialize eviction policy
        self._eviction_policy = self._create_eviction_policy()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
    def get(self, key: Any) -> Optional[T]:
        """
        Get value from cache.
        
        Parameters
        ----------
        key : Any
            Cache key
            
        Returns
        -------
        Optional[T]
            Cached value or None if not found
        """
        with self._lock:
            # Check main cache first
            if key in self._cache:
                entry = self._cache[key]
                
                # Check if expired
                if entry.is_expired():
                    del self._cache[key]
                    self._misses += 1
                    return None
                    
                # Update access metadata
                entry.touch()
                self._hits += 1
                return entry.value
                
            # Check weak reference cache
            if self._weak_cache and key in self._weak_cache:
                weak_ref = self._weak_cache[key]
                value = weak_ref()
                
                if value is not None:
                    # Promote back to main cache
                    self._put_internal(key, value, self._default_ttl)
                    self._hits += 1
                    return value
                else:
                    # Reference was garbage collected
                    del self._weak_cache[key]
                    
            self._misses += 1
            return None
            
    def put(self, key: Any, value: T, ttl: Optional[float] = None) -> None:
        """
        Put value in cache.
        
        Parameters
        ----------
        key : Any
            Cache key
        value : T
            Value to cache
        ttl : float, optional
            TTL in seconds (uses default if not provided)
        """
        with self._lock:
            self._put_internal(key, value, ttl)
            
    def _put_internal(self, key: Any, value: T, ttl: Optional[float]) -> None:
        """Internal put implementation."""
        # Remove existing entry if present
        if key in self._cache:
            del self._cache[key]
            
        # Check if we need to evict
        if len(self._cache) >= self._max_size:
            self._evict_entries(1)
            
        # Create cache entry
        entry = CacheEntry(
            value=value,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            ttl_seconds=ttl or self._default_ttl,
        )
        
        self._cache[key] = entry
        
        # Also store weak reference if enabled
        if self._weak_cache:
            try:
                self._weak_cache[key] = weakref.ref(value)
            except TypeError:
                # Object doesn't support weak references
                pass
                
    def invalidate(self, key: Any) -> bool:
        """
        Invalidate cache entry.
        
        Parameters
        ----------
        key : Any
            Cache key to invalidate
            
        Returns
        -------
        bool
            True if entry was removed
        """
        with self._lock:
            removed = key in self._cache
            if removed:
                del self._cache[key]
                
            if self._weak_cache and key in self._weak_cache:
                del self._weak_cache[key]
                
            return removed
            
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            if self._weak_cache:
                self._weak_cache.clear()
            self._logger.debug("Cache cleared")
            
    def cleanup_expired(self) -> int:
        """
        Remove expired entries.
        
        Returns
        -------
        int
            Number of entries removed
        """
        with self._lock:
            current_time = time.time()
            expired_keys = []
            
            for key, entry in self._cache.items():
                if self._eviction_policy.should_evict(entry, current_time):
                    expired_keys.append(key)
                    
            for key in expired_keys:
                del self._cache[key]
                
            self._evictions += len(expired_keys)
            
            if expired_keys:
                self._logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
                
            return len(expired_keys)
            
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns
        -------
        Dict[str, Any]
            Cache statistics
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_ratio = self._hits / total_requests if total_requests > 0 else 0.0
            
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio": hit_ratio,
                "evictions": self._evictions,
                "size": len(self._cache),
                "max_size": self._max_size,
                "strategy": self._strategy.value,
                "weak_cache_size": len(self._weak_cache) if self._weak_cache else 0,
            }
            
    def resize(self, new_max_size: int) -> None:
        """
        Resize cache.
        
        Parameters
        ----------
        new_max_size : int
            New maximum cache size
        """
        with self._lock:
            old_size = self._max_size
            self._max_size = new_max_size
            
            # Evict if necessary
            if len(self._cache) > new_max_size:
                evict_count = len(self._cache) - new_max_size
                self._evict_entries(evict_count)
                
            self._logger.info(f"Cache resized from {old_size} to {new_max_size}")
            
    def _evict_entries(self, count: int) -> None:
        """Evict entries using the configured policy."""
        if count <= 0 or not self._cache:
            return
            
        candidates = self._eviction_policy.get_eviction_candidates(self._cache, count)
        
        for key in candidates:
            if key in self._cache:
                del self._cache[key]
                self._evictions += 1
                
        if candidates:
            self._logger.debug(f"Evicted {len(candidates)} entries using {self._strategy.value} policy")
            
    def _create_eviction_policy(self) -> CacheEvictionPolicy:
        """Create eviction policy based on strategy."""
        if self._strategy == CacheStrategy.LRU:
            return LRUEvictionPolicy()
        elif self._strategy == CacheStrategy.LFU:
            return LFUEvictionPolicy()
        elif self._strategy == CacheStrategy.TTL:
            return TTLEvictionPolicy(self._default_ttl or 300.0)
        else:
            return LRUEvictionPolicy()  # Default fallback


class CachedServiceProvider:
    """
    Service provider wrapper that adds caching capabilities.
    """
    
    def __init__(
        self,
        original_provider: Any,
        cache: ServiceCache,
        cache_key_func: Optional[callable] = None,
    ) -> None:
        """
        Initialize cached service provider.
        
        Parameters
        ----------
        original_provider : Any
            Original service provider
        cache : ServiceCache
            Cache instance
        cache_key_func : callable, optional
            Function to generate cache keys
        """
        self.original_provider = original_provider
        self.cache = cache
        self.cache_key_func = cache_key_func or self._default_cache_key
        self._logger = Logger(self.__class__.__name__)
        
    def get(self, container: Any, resolution_chain: Optional[Set[Type]] = None) -> Any:
        """
        Get service with caching.
        
        Parameters
        ----------
        container : Any
            DI container
        resolution_chain : Set[Type], optional
            Current resolution chain
            
        Returns
        -------
        Any
            Service instance
        """
        # Generate cache key
        cache_key = self.cache_key_func(self.original_provider, resolution_chain)
        
        # Check cache first
        cached_value = self.cache.get(cache_key)
        if cached_value is not None:
            # Record cache hit for monitoring
            if hasattr(container, '_metrics_collector') and container._metrics_collector:
                container._metrics_collector.record_cache_hit(
                    self.original_provider.descriptor.interface.__name__, True
                )
            return cached_value
            
        # Cache miss - get from original provider
        value = self.original_provider.get(container, resolution_chain)
        
        # Cache the result (only for singletons and scoped services)
        lifetime = self.original_provider.descriptor.lifetime.value
        if lifetime in ["singleton", "scoped"]:
            ttl = None  # No TTL for singleton/scoped
            if lifetime == "scoped":
                ttl = 3600.0  # 1 hour TTL for scoped services
                
            self.cache.put(cache_key, value, ttl)
            
        # Record cache miss for monitoring
        if hasattr(container, '_metrics_collector') and container._metrics_collector:
            container._metrics_collector.record_cache_hit(
                self.original_provider.descriptor.interface.__name__, False
            )
            
        return value
        
    def _default_cache_key(self, provider: Any, resolution_chain: Optional[Set[Type]]) -> str:
        """Generate default cache key."""
        interface_name = provider.descriptor.interface.__name__
        lifetime = provider.descriptor.lifetime.value
        
        # For scoped services, include resolution chain in key
        if lifetime == "scoped" and resolution_chain:
            chain_key = "_".join(sorted(t.__name__ for t in resolution_chain))
            return f"{interface_name}_{lifetime}_{chain_key}"
        else:
            return f"{interface_name}_{lifetime}"


class CacheManager:
    """
    Manages multiple caches and provides cache coordination.
    """
    
    def __init__(self) -> None:
        """Initialize cache manager."""
        self._caches: Dict[str, ServiceCache] = {}
        self._logger = Logger(self.__class__.__name__)
        self._cleanup_thread: Optional[threading.Thread] = None
        self._cleanup_interval = 60.0  # 1 minute
        self._running = False
        
    def create_cache(
        self,
        name: str,
        max_size: int = 1000,
        strategy: CacheStrategy = CacheStrategy.LRU,
        default_ttl: Optional[float] = None,
        enable_weak_refs: bool = False,
    ) -> ServiceCache:
        """
        Create a new cache.
        
        Parameters
        ----------
        name : str
            Cache name
        max_size : int
            Maximum cache size
        strategy : CacheStrategy
            Eviction strategy
        default_ttl : float, optional
            Default TTL in seconds
        enable_weak_refs : bool
            Use weak references
            
        Returns
        -------
        ServiceCache
            Created cache
        """
        cache = ServiceCache(
            max_size=max_size,
            strategy=strategy,
            default_ttl=default_ttl,
            enable_weak_refs=enable_weak_refs,
        )
        
        self._caches[name] = cache
        self._logger.info(f"Created cache '{name}' with strategy {strategy.value}")
        
        return cache
        
    def get_cache(self, name: str) -> Optional[ServiceCache]:
        """Get cache by name."""
        return self._caches.get(name)
        
    def clear_all_caches(self) -> None:
        """Clear all managed caches."""
        for cache in self._caches.values():
            cache.clear()
        self._logger.info("Cleared all caches")
        
    def get_cache_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all caches."""
        return {name: cache.get_stats() for name, cache in self._caches.items()}
        
    def start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            return
            
        self._running = True
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="CacheCleanup"
        )
        self._cleanup_thread.start()
        self._logger.info("Started cache cleanup thread")
        
    def stop_cleanup_thread(self) -> None:
        """Stop background cleanup thread."""
        self._running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)
        self._logger.info("Stopped cache cleanup thread")
        
    def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while self._running:
            try:
                for name, cache in self._caches.items():
                    expired_count = cache.cleanup_expired()
                    if expired_count > 0:
                        self._logger.debug(f"Cache '{name}': cleaned up {expired_count} expired entries")
                        
                # Sleep with interruption check
                for _ in range(int(self._cleanup_interval)):
                    if not self._running:
                        break
                    time.sleep(1.0)
                    
            except Exception as e:
                self._logger.error(f"Error in cache cleanup thread: {e}")
                time.sleep(5.0)  # Brief pause before continuing


# Global cache manager
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def create_default_cache() -> ServiceCache:
    """Create default service cache."""
    return ServiceCache(
        max_size=1000,
        strategy=CacheStrategy.LRU,
        default_ttl=300.0,  # 5 minutes
        enable_weak_refs=True,
    )