"""
Legacy shim — contents merged into scheduler_base.py.
Import from scheduler_base directly.
"""
from .scheduler_base import Scheduler, is_notebook

__all__ = ["Scheduler", "is_notebook"]
