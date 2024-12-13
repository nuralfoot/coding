"""Covers feed forward and back prop neurodes for training.

Exploits multiple inheritance to inherit methods from both types of neurode.
"""
from FFNeurode import FFNeurode
from BPNeurode import BPNeurode


class FFBPNeurode(FFNeurode, BPNeurode):
    """FFNeurode first for method order."""

    pass
