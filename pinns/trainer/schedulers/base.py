"""
Base learning rate scheduler and utilities.
"""

import numpy as np
from abc import ABC, abstractmethod


def is_notebook():
    """Check if running in a Jupyter notebook."""
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal IPython
        else:
            return False
    except (NameError, AttributeError):
        return False


class LRScheduler(ABC):
    """
    Abstract base class for learning rate schedulers.
    
    Subclasses must implement the `lr` method that computes
    the new learning rate given the base learning rate and current step.
    """
    
    @abstractmethod
    def lr(self, base_lr: float, step: int) -> float:
        """
        Compute the learning rate for a given step.
        
        Args:
            base_lr: The initial/base learning rate
            step: Current training step (epoch)
            
        Returns:
            The adjusted learning rate
        """
        pass


__all__ = ["is_notebook", "LRScheduler"]
