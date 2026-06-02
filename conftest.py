"""
Pytest configuration for starfish tests.

This module provides fixtures and hooks for pytest to properly manage resources
during test execution, particularly for Python 3.13 compatibility where stricter
resource management warnings are enabled.
"""
import gc
import pytest
import warnings


# Filter ResourceWarning for unclosed databases from diskcache
# This is a known issue with the slicedimage library's use of diskcache
# where Cache objects are stored in a class-level cache and not explicitly closed
warnings.filterwarnings(
    "ignore",
    message="unclosed.*database",
    category=ResourceWarning,
)


def _cleanup_diskcache_caches():
    """
    Helper function to close all diskcache Cache objects from slicedimage.
    
    In Python 3.13, stricter ResourceWarnings are issued for unclosed database
    connections. The slicedimage library uses diskcache.Cache objects that are
    stored in a class-level cache and never explicitly closed. This function
    ensures they are properly closed.
    """
    try:
        from slicedimage.backends._caching import CachingBackend
        if hasattr(CachingBackend, '_CACHE'):
            for cache in CachingBackend._CACHE.values():
                try:
                    cache.close()
                except Exception:
                    # Ignore errors during cleanup
                    pass
            CachingBackend._CACHE.clear()
    except ImportError:
        # slicedimage not installed or structure changed
        pass


@pytest.fixture(scope="function", autouse=True)
def cleanup_diskcache_per_test():
    """
    Fixture to clean up diskcache Cache objects after each test function.
    
    This ensures that Cache objects don't accumulate and trigger ResourceWarnings
    during garbage collection in Python 3.13.
    """
    yield
    _cleanup_diskcache_caches()
    # Force garbage collection after cleanup to prevent warnings
    gc.collect()


@pytest.fixture(scope="session", autouse=True)
def cleanup_diskcache_session():
    """
    Fixture to clean up diskcache Cache objects at the end of the test session.
    
    This is a safety net to ensure all Cache objects are closed even if the
    per-test cleanup misses any.
    """
    yield
    _cleanup_diskcache_caches()