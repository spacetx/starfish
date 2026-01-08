"""
Pytest configuration for starfish tests.

This module provides fixtures and hooks for pytest to properly manage resources
during test execution, particularly for Python 3.13 compatibility where stricter
resource management warnings are enabled.
"""
import pytest


@pytest.fixture(scope="session", autouse=True)
def cleanup_diskcache():
    """
    Fixture to clean up diskcache Cache objects after all tests complete.
    
    In Python 3.13, stricter ResourceWarnings are issued for unclosed database
    connections. The slicedimage library uses diskcache.Cache objects that are
    stored in a class-level cache and never explicitly closed. This fixture
    ensures they are properly closed at the end of the test session.
    """
    yield
    
    # Clean up diskcache Cache objects from slicedimage
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
