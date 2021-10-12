from typing import Any, Callable, Dict

import funcy

__all__ = ["pre_processing"]

def pre_processing(options: Dict[str, Any]) -> Callable:
    return funcy.identity