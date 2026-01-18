"""
Caching utilities matching vace_hybrid.ipynb implementation.

Uses absolute step numbers for start/end, with interval-based caching.
This is NOT the same as cache-dit library - this is our simple built-in caching
that stores the noise prediction from the previous step and reuses it.
"""

from typing import Optional


def should_use_cache(
    global_step: int,
    start_step: int,
    end_step: Optional[int],
    interval: int
) -> bool:
    """
    Determine if we should use cached result for this step.
    
    This implements the exact same logic as vace_hybrid.ipynb:
    - Before start_step: always compute fresh
    - After end_step: always compute fresh
    - Within caching zone: fresh on interval boundaries, cache otherwise
    
    Args:
        global_step: Current step number (0-indexed)
        start_step: Absolute step to start caching (e.g., 10)
        end_step: Absolute step to stop caching (e.g., 40), None = cache until end
        interval: Cache interval - compute fresh every N steps
    
    Returns:
        True = use cached result (skip computation)
        False = compute fresh (and store in cache)
    
    Example:
        With start_step=10, end_step=40, interval=3:
        - Step 9: False (before start)
        - Step 10: False (fresh - 0 % 3 == 0)
        - Step 11: True (cache - 1 % 3 != 0)
        - Step 12: True (cache - 2 % 3 != 0)
        - Step 13: False (fresh - 3 % 3 == 0)
        - Step 41: False (after end)
    """
    # Before start_step: always compute fresh
    if global_step < start_step:
        return False
    
    # After end_step: always compute fresh
    if end_step is not None and global_step > end_step:
        return False
    
    # Within caching zone: check interval
    # Fresh steps: steps_since_start % interval == 0 (0, 3, 6, 9, ...)
    # Cache steps: steps_since_start % interval != 0 (1, 2, 4, 5, 7, 8, ...)
    steps_since_start = global_step - start_step
    return steps_since_start % interval != 0


def is_fresh_step(
    global_step: int,
    start_step: int,
    end_step: Optional[int],
    interval: int
) -> bool:
    """
    Inverse of should_use_cache - returns True if we need to compute fresh.
    
    This is the step where we:
    1. Run the actual model inference
    2. Store the result in cache for subsequent steps
    """
    return not should_use_cache(global_step, start_step, end_step, interval)


def get_cache_statistics(
    total_steps: int,
    start_step: int,
    end_step: Optional[int],
    interval: int
) -> dict:
    """
    Calculate expected cache statistics for given configuration.
    
    Useful for:
    - Estimating speedup before generation
    - Reporting statistics after generation
    
    Args:
        total_steps: Total number of sampling steps
        start_step: Absolute step to start caching
        end_step: Absolute step to stop caching (None = until end)
        interval: Cache interval
    
    Returns:
        Dictionary with:
        - total_steps: Total steps
        - cache_hits: Number of steps using cache
        - fresh_computes: Number of steps computing fresh
        - cache_hit_rate: Ratio of cache hits (0.0 - 1.0)
        - estimated_speedup: Rough speedup estimate
    """
    cache_hits = 0
    fresh_computes = 0
    
    actual_end = end_step if end_step is not None else total_steps - 1
    
    for step in range(total_steps):
        if should_use_cache(step, start_step, actual_end, interval):
            cache_hits += 1
        else:
            fresh_computes += 1
    
    hit_rate = cache_hits / total_steps if total_steps > 0 else 0.0
    
    # Speedup estimate: if we skip 40% of computations, we're ~1.4x faster
    # This is approximate since cache hits aren't completely free
    estimated_speedup = total_steps / fresh_computes if fresh_computes > 0 else 1.0
    
    return {
        "total_steps": total_steps,
        "cache_hits": cache_hits,
        "fresh_computes": fresh_computes,
        "cache_hit_rate": round(hit_rate, 3),
        "estimated_speedup": round(estimated_speedup, 2)
    }


def validate_cache_config(
    total_steps: int,
    start_step: int,
    end_step: Optional[int],
    interval: int
) -> tuple[bool, str]:
    """
    Validate caching configuration.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if start_step < 0:
        return False, "cache_start_step must be >= 0"
    
    if start_step >= total_steps:
        return False, f"cache_start_step ({start_step}) must be < total_steps ({total_steps})"
    
    if end_step is not None:
        if end_step < start_step:
            return False, f"cache_end_step ({end_step}) must be >= cache_start_step ({start_step})"
        if end_step >= total_steps:
            return False, f"cache_end_step ({end_step}) must be < total_steps ({total_steps})"
    
    if interval < 2:
        return False, "cache_interval must be >= 2 (1 would mean no caching)"
    
    if interval > total_steps:
        return False, f"cache_interval ({interval}) should be <= total_steps ({total_steps})"
    
    return True, ""

