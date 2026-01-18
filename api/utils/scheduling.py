"""
Scheduling utilities for hybrid model sampling.

Handles parsing, validation, and step-to-model mapping for hybrid schedules.
"""

from typing import Optional


def validate_schedule(schedule: list, total_steps: int) -> tuple[bool, str]:
    """
    Validate a sampling schedule.
    
    Args:
        schedule: List of [model_name, num_steps] pairs
                  e.g., [["14B", 15], ["1.3B", 35]]
        total_steps: Total sampling steps (must match sum of schedule)
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not schedule:
        return False, "Schedule cannot be empty"
    
    valid_models = {"14B", "1.3B"}
    total_scheduled = 0
    
    for i, segment in enumerate(schedule):
        # Check format
        if not isinstance(segment, (list, tuple)) or len(segment) != 2:
            return False, f"Segment {i} must be [model_name, num_steps], got {segment}"
        
        model_name, num_steps = segment
        
        # Check model name
        if model_name not in valid_models:
            return False, f"Invalid model '{model_name}' in segment {i}. Must be one of {valid_models}"
        
        # Check steps
        if not isinstance(num_steps, int) or num_steps <= 0:
            return False, f"num_steps must be positive integer, got {num_steps} in segment {i}"
        
        total_scheduled += num_steps
    
    # Check total
    if total_scheduled != total_steps:
        return False, f"Schedule steps ({total_scheduled}) must equal total_steps ({total_steps})"
    
    return True, ""


def parse_schedule(schedule: list) -> list[tuple[str, int]]:
    """
    Convert schedule from JSON format to internal tuple format.
    
    Args:
        schedule: List of [model_name, num_steps] pairs
                  e.g., [["14B", 15], ["1.3B", 35]]
    
    Returns:
        List of (model_name, num_steps) tuples
        e.g., [("14B", 15), ("1.3B", 35)]
    """
    return [(str(segment[0]), int(segment[1])) for segment in schedule]


def get_model_for_step(step: int, schedule: list[tuple[str, int]]) -> str:
    """
    Determine which model handles a specific step.
    
    Args:
        step: Step number (0-indexed)
        schedule: Parsed schedule as list of (model_name, num_steps) tuples
    
    Returns:
        Model name ("14B" or "1.3B")
    
    Example:
        schedule = [("14B", 15), ("1.3B", 35)]
        get_model_for_step(0, schedule)  -> "14B"
        get_model_for_step(14, schedule) -> "14B"
        get_model_for_step(15, schedule) -> "1.3B"
        get_model_for_step(49, schedule) -> "1.3B"
    """
    cumulative = 0
    for model_name, num_steps in schedule:
        cumulative += num_steps
        if step < cumulative:
            return model_name
    
    # If step is beyond schedule, return last model
    return schedule[-1][0] if schedule else "14B"


def get_default_schedule(total_steps: int) -> list[tuple[str, int]]:
    """
    Generate default hybrid schedule for given total steps.
    
    Default strategy: 30% 14B at start, 70% 1.3B for the rest
    This balances quality (14B sets structure) with speed (1.3B refines)
    
    Args:
        total_steps: Total number of sampling steps
    
    Returns:
        Schedule as list of (model_name, num_steps) tuples
    """
    # 30% with 14B, 70% with 1.3B
    steps_14B = max(1, int(total_steps * 0.3))
    steps_1_3B = total_steps - steps_14B
    
    return [("14B", steps_14B), ("1.3B", steps_1_3B)]


def get_schedule_summary(schedule: list[tuple[str, int]]) -> str:
    """
    Generate human-readable schedule summary.
    
    Args:
        schedule: Parsed schedule
    
    Returns:
        Summary string like "14B(15) → 1.3B(35)"
    """
    return " → ".join([f"{model}({steps})" for model, steps in schedule])


def get_segment_boundaries(schedule: list[tuple[str, int]]) -> list[dict]:
    """
    Calculate step boundaries for each segment.
    
    Args:
        schedule: Parsed schedule
    
    Returns:
        List of segment info dicts with start_step, end_step, model, num_steps
    """
    boundaries = []
    current_step = 0
    
    for model_name, num_steps in schedule:
        boundaries.append({
            "model": model_name,
            "num_steps": num_steps,
            "start_step": current_step,
            "end_step": current_step + num_steps - 1
        })
        current_step += num_steps
    
    return boundaries

