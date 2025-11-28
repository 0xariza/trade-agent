"""
Trading module for Alpha Arena.

Contains spot trading logic.
"""

from .spot_signal_handler import (
    SpotSignalHandler,
    SpotAction,
    SignalDecision,
    process_spot_signal
)

__all__ = [
    "SpotSignalHandler",
    "SpotAction", 
    "SignalDecision",
    "process_spot_signal"
]

