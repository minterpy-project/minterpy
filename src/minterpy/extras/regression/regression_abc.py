"""
This module contains the abstract base class for polynomial regression classes.

All concrete implementations of the polynomial regression classes must inherit
from this abstract base class.
This abstract class ensures that all polynomial regression classes share
a common interface, while allowing for variations in their implementation
and the addition of specific methods or attributes.
"""

import abc


__all__ = ["RegressionABC"]


class RegressionABC(abc.ABC):
    """The abstract base class for all regression models."""

    @abc.abstractmethod
    def fit(self, xx, yy, *args, **kwargs):  # pragma: no cover
        """Abstract container for fitting a polynomial regression."""
        pass

    @abc.abstractmethod
    def predict(self, xx):  # pragma: no cover
        """Abstract container for making prediction using a polynomial regression."""
        pass

    @abc.abstractmethod
    def show(self):  # pragma: no cover
        """Abstract container for printing out the details of a polynomial regression model."""
        pass

    def __call__(self, xx):  # pragma: no cover
        """Evaluation of the polynomial regression model."""
        return self.predict(xx)
