"""Signal processing utilities for smoothing policy outputs."""

from collections import deque
import time
from typing import Any

import numpy as np


class FIRFilter:
    """Simple FIR (Finite Impulse Response) filter for numpy arrays in dictionaries.

    Maintains a rolling buffer of the last k inputs and applies exponentially-weighted
    averaging based on the alpha parameter.
    """

    def __init__(self, *, k: int, alpha: float):
        """Initialize FIR filter.

        Args:
            k: Buffer size (number of samples to keep in the rolling buffer)
            alpha: Smoothing factor for exponential weights (0 < alpha <= 1).
                   Values closer to 1 give more weight to recent samples.
        """
        self.k = k
        self.alpha = alpha
        self._buffer: dict[str, deque[np.ndarray]] = {}
        self._array_keys: set[str] | None = None

    def step(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply FIR filter to numpy arrays in the input dictionary.

        Args:
            data: Dictionary containing values, some of which may be numpy arrays

        Returns:
            Dictionary with filtered numpy arrays and original non-array values

        Raises:
            ValueError: If the set of keys containing numpy arrays changes after
                       the first step() call
        """
        # Identify keys with numpy arrays
        current_array_keys = {k for k, v in data.items() if isinstance(v, np.ndarray)}

        # Check if array keys have changed
        if self._array_keys is None:
            self._array_keys = current_array_keys
            # Initialize buffer for each array key
            for key in current_array_keys:
                self._buffer[key] = deque(maxlen=self.k)
        elif current_array_keys != self._array_keys:
            raise ValueError(
                f"Keys containing numpy arrays have changed. "
                f"Expected {self._array_keys}, got {current_array_keys}"
            )

        # Add to buffers (all buffers will have the same length)
        for key in self._array_keys:
            self._buffer[key].append(data[key].copy())

        # Compute exponential weights once (all buffers have same length)
        n = len(self._buffer[next(iter(self._array_keys))])
        weights = np.array([self.alpha ** (n - 1 - i) for i in range(n)])
        weights = weights / weights.sum()  # Normalize

        # Apply weighted average to buffered samples
        result = {}
        for key, value in data.items():
            if key in self._array_keys:
                filtered = np.zeros_like(value, dtype=float)
                for i, buffered_value in enumerate(self._buffer[key]):
                    filtered += weights[i] * buffered_value
                result[key] = filtered
            else:
                # Pass through non-array values
                result[key] = value

        return result


class PeriodicAverageAccumulator:
    """Accumulates values and computes averages over a time window.

    Useful for periodic logging of metrics without spamming on every iteration.
    """

    def __init__(self, window_seconds: float = 1.0):
        """Initialize the accumulator.

        Args:
            window_seconds: Time window in seconds for computing averages.
        """
        self.window_seconds = window_seconds
        self._values: list[float] = []
        self._window_start: float | None = None

    def add(self, value: float) -> float | None:
        """Add a value to the accumulator.

        Args:
            value: The value to accumulate.

        Returns:
            The average over the window if the window has elapsed, None otherwise.
        """
        now = time.monotonic()

        if self._window_start is None:
            self._window_start = now

        self._values.append(value)

        if now - self._window_start >= self.window_seconds:
            avg = sum(self._values) / len(self._values) if self._values else 0.0
            self._values.clear()
            self._window_start = now
            return avg

        return None

    def reset(self) -> None:
        """Reset the accumulator."""
        self._values.clear()
        self._window_start = None
