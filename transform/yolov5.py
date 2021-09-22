from typing import Any, Callable, Dict

from torchvision import transforms as T
import funcy

__all__ = ["pre_processing"]

def pre_processing(options: Dict[str, Any]) -> Callable:
    return funcy.identity

