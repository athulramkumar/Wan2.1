"""
Distributed hybrid inference for Wan2.1.

Splits generation across a coordinator (stable GPU, 1.3B model) and
a worker (spot instance, 14B model) with checkpoint-based segment delegation.
"""
