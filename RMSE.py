"""
RMSE calculations for implementing distances.

Supports both Euclidean and Taxicab distances.
"""

from abc import ABC, abstractmethod
from math import sqrt


class RMSE(ABC):
    """Parent class defines how RMSE should work."""

    def __init__(self):
        """Create an empty list to store predicted and actual value."""
        self._data = []

    def __add__(self, points):
        """Add predicted and expected points and return new instance."""
        predicted, expected = points
        if len(predicted) != len(expected):
            raise ValueError("Points must have same dimensions")

        new_obj = self.__class__()
        new_obj._data = self._data.copy()
        new_obj._data.append((predicted, expected))
        return new_obj

    def __iadd__(self, points):
        """Add predicted and expected points in place."""
        predicted, expected = points
        if len(predicted) != len(expected):
            raise ValueError("Points must have same dimensions")

        self._data.append((predicted, expected))
        return self

    def reset(self):
        """Start over with empty data."""
        self._data = []

    @property
    def error(self):
        """
        Calculate the total RMSE employing whichever of the distance formulas.

        Returns 0 if empty.
        """
        if not self._data:
            return 0

        squared_error_sum = 0
        for predicted, expected in self._data:
            dist = self.distance(predicted, expected)
            squared_error_sum += dist * dist

        mean_squared_error = squared_error_sum / len(self._data)
        return sqrt(mean_squared_error)

    @staticmethod
    @abstractmethod
    def distance(predicted, expected):
        """Find difference between predicted and expected point."""
        pass


class Euclidean(RMSE):
    """Implements RMSE using Euclidean distance."""

    @staticmethod
    def distance(predicted, expected):
        """Calculate Euclidean distance between two points."""
        return sqrt(sum((p - e) ** 2 for p, e in zip(predicted, expected)))


class Taxicab(RMSE):
    """Uses Taxicab distance for the RMSE."""

    @staticmethod
    def distance(predicted, expected):
        """Compute Taxicab distance of two points."""
        return sum(abs(p - e) for p, e in zip(predicted, expected))
