"""
ReduceLROnPlateau learning rate scheduler.
"""

from .base import LRScheduler


class ReduceLROnPlateau(LRScheduler):
    """
    Reduce learning rate when training loss plateaus.
    
    Uses exponential moving average (EMA) of the loss and detects plateau by
    computing the relative slope over a window:
    
        relative_slope = |EMA_t - EMA_{t-W}| / EMA_{t-W}
    
    If relative_slope < epsilon, reduce LR by factor.
    
    Example:
        scheduler = ReduceLROnPlateau(window=1000, epsilon=1e-3, factor=0.5)
        trainer.compile(lr_scheduler=scheduler, ...)
        
    Attributes:
        window: Number of steps to compute slope over (default: 1000)
        epsilon: Threshold for relative change to trigger reduction (default: 1e-3)
        factor: Factor to multiply LR when plateau detected (default: 0.5)
        ema_alpha: Smoothing factor for EMA (default: 0.99)
        min_lr: Minimum learning rate (default: 1e-8)
        cooldown: Steps to wait after reduction before checking again (default: same as window)
    """
    
    def __init__(self, 
                 window: int = 1000, 
                 epsilon: float = 1e-3, 
                 factor: float = 0.5,
                 ema_alpha: float = 0.99,
                 min_lr: float = 1e-8,
                 cooldown: int = None):
        """
        Initialize ReduceLROnPlateau scheduler.
        
        Args:
            window: Number of steps to compute slope over. Larger = less sensitive to noise.
            epsilon: Threshold for relative change. If |slope| < epsilon, plateau detected.
            factor: Multiply LR by this when plateau detected (0 < factor < 1).
            ema_alpha: EMA smoothing factor. Higher = smoother (0 < alpha < 1).
            min_lr: Minimum learning rate floor.
            cooldown: Steps to wait after reduction before checking again.
        """
        if window <= 0:
            raise ValueError("window must be positive")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not 0 < factor < 1:
            raise ValueError("factor must be between 0 and 1")
        if not 0 < ema_alpha < 1:
            raise ValueError("ema_alpha must be between 0 and 1")
        
        self.window = window
        self.epsilon = epsilon
        self.factor = factor
        self.ema_alpha = ema_alpha
        self.min_lr = min_lr
        self.cooldown = cooldown if cooldown is not None else window
        
        # State
        self._ema = None  # Current EMA
        self._ema_history = []  # History of EMA values (circular buffer)
        self._reduction_count = 0  # Number of times LR has been reduced
        self._last_reduction_step = -float('inf')  # Step of last reduction
        self._current_step = 0
    
    def step(self, loss: float, current_step: int = None):
        """
        Update the scheduler with the current loss value.
        
        Call this after each training step with the current loss.
        
        Args:
            loss: Current training loss
            current_step: Current step number (optional, auto-incremented if None)
        """
        if current_step is not None:
            self._current_step = current_step
        else:
            self._current_step += 1
        
        # Update EMA
        if self._ema is None:
            self._ema = loss
        else:
            self._ema = self.ema_alpha * self._ema + (1 - self.ema_alpha) * loss
        
        # Store in history
        self._ema_history.append(self._ema)
        
        # Keep only window + 1 entries
        if len(self._ema_history) > self.window + 1:
            self._ema_history.pop(0)
        
        # Check for plateau if we have enough history and past cooldown
        if len(self._ema_history) > self.window:
            steps_since_reduction = self._current_step - self._last_reduction_step
            
            if steps_since_reduction >= self.cooldown:
                ema_old = self._ema_history[0]
                ema_new = self._ema_history[-1]
                
                # Compute relative slope
                if ema_old > 1e-10:  # Avoid division by zero
                    relative_slope = abs(ema_new - ema_old) / ema_old
                    
                    if relative_slope < self.epsilon:
                        # Plateau detected - reduce LR
                        self._reduction_count += 1
                        self._last_reduction_step = self._current_step
    
    def lr(self, base_lr: float, step: int) -> float:
        """
        Compute the current learning rate.
        
        Args:
            base_lr: The initial/base learning rate
            step: Current training step (epoch)
            
        Returns:
            Learning rate after reductions
        """
        new_lr = base_lr * (self.factor ** self._reduction_count)
        return max(new_lr, self.min_lr)
    
    def reset(self):
        """Reset scheduler state."""
        self._ema = None
        self._ema_history = []
        self._reduction_count = 0
        self._last_reduction_step = -float('inf')
        self._current_step = 0


__all__ = ["ReduceLROnPlateau"]
