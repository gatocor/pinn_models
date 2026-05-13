"""
Exponential decay learning rate scheduler.
"""

from .scheduler_base import Scheduler


class SchedulerExponentialDecay(Scheduler):
    """
    Exponential decay learning rate scheduler.
    
    Reduces learning rate by a factor of gamma every N steps.
    
    Formula:
        new_lr = base_lr * gamma^(step // each_n_steps)
    
    Example:
        scheduler = ExponentialDecay(gamma=0.9, each_n_steps=1000)
        trainer.compile(lr_scheduler=scheduler, ...)
        
        # At step 0: lr = base_lr * 0.9^0 = base_lr
        # At step 1000: lr = base_lr * 0.9^1 = 0.9 * base_lr
        # At step 2000: lr = base_lr * 0.9^2 = 0.81 * base_lr
    
    Attributes:
        gamma: Decay factor (0 < gamma < 1 for decay)
        each_n_steps: Number of steps between decay applications
    """
    
    def __init__(self, gamma: float = 0.9, each_n_steps: int = 1000):
        """
        Initialize exponential decay scheduler.
        
        Args:
            gamma: Decay factor. Learning rate is multiplied by gamma 
                   every each_n_steps. Typical values: 0.9, 0.95, 0.99.
            each_n_steps: Number of steps between each decay application.
        """
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if each_n_steps <= 0:
            raise ValueError("each_n_steps must be positive")
        
        self.gamma = gamma
        self.each_n_steps = each_n_steps
    
    def lr(self, base_lr: float, step: int) -> float:
        """
        Compute learning rate with exponential decay.
        
        Args:
            base_lr: The initial/base learning rate
            step: Current training step (epoch)
            
        Returns:
            new_lr = base_lr * gamma^(step // each_n_steps)
        """
        decay_count = step // self.each_n_steps
        return base_lr * (self.gamma ** decay_count)


__all__ = ["SchedulerExponentialDecay"]
