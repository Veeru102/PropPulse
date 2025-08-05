from typing import Any, Dict, Optional
import time
from threading import Lock

class InMemoryCache:
    """Thread-safe in-memory cache with TTL support"""
    
    def __init__(self):
        self._cache: Dict[str, tuple[Any, float]] = {}  # (value, expiry)
        self._lock = Lock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if it exists and hasn't expired"""
        with self._lock:
            if key not in self._cache:
                return None
                
            value, expiry = self._cache[key]
            if expiry < time.time():
                del self._cache[key]
                return None
                
            return value
            
    def set(self, key: str, value: Any, ttl: int = 86400) -> None:
        """Set value in cache with TTL in seconds (default 24h)"""
        with self._lock:
            expiry = time.time() + ttl
            self._cache[key] = (value, expiry)
            
    def delete(self, key: str) -> None:
        """Remove key from cache"""
        with self._lock:
            self._cache.pop(key, None)
            
    def clear(self) -> None:
        """Clear all cached items"""
        with self._lock:
            self._cache.clear()
            
    def cleanup(self) -> None:
        """Remove expired entries"""
        with self._lock:
            now = time.time()
            expired = [k for k, (_, exp) in self._cache.items() if exp < now]
            for k in expired:
                del self._cache[k]

# Global cache instances for different types of data
property_cache = InMemoryCache()  # 24h TTL for property data
market_cache = InMemoryCache()    # 1h TTL for market data
comparable_cache = InMemoryCache()  # 24h TTL for comparable properties