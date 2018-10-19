import importlib
from typing import Any

def try_import(module: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError:
        raise ImportError(
            f"{module} is an optional dependency of starfish. Please install {module} and its "
            f"dependenciesto use this functionality."
        )
