"""Processor base for processing stages."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Processor(ABC):
    """Abstract base class for processing stages.

    Subclasses must implement `apply(data)`.
    """

    @abstractmethod
    def apply(self, data: Any) -> Any:
        """Apply processing to `data` and return the result."""

        raise NotImplementedError
