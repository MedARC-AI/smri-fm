from __future__ import annotations

from typing import Protocol

from torch import Tensor


class Backbone(Protocol):
    """Backbone contract used by probe evaluations."""

    def encode(self, batch: Tensor) -> Tensor:
        """Return one embedding vector per input sample."""
