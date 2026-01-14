# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Acceleration utilities for Wan video generation models.

This module provides integration with:
- cache-dit: Activation caching for faster inference
- xDiT/xFuser: Distributed inference with sequence parallelism

Usage:
    from wan.utils.acceleration import (
        setup_cache_dit,
        setup_xdit,
        AccelerationConfig,
    )
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal

import torch

logger = logging.getLogger(__name__)


@dataclass
class CacheDiTConfig:
    """Configuration for cache-dit acceleration.

    Cache-DiT caches intermediate DiT activations to skip redundant computations
    during the diffusion sampling process.

    Attributes:
        enabled: Whether to enable cache-dit acceleration
        cache_type: Type of caching strategy
            - 'db': DBCache (recommended for most cases)
            - 'taylor': TaylorSeer cache
            - 'scm': SCM cache
        cache_interval: Cache update interval (higher = more aggressive caching)
        cache_start_step: Start caching after this many steps (allow model to stabilize)
        cache_end_step: Stop caching before this many steps from the end (optional)
        fresh_ratio: Ratio of steps to compute fresh (0.0-1.0, for adaptive caching)
    """
    enabled: bool = False
    cache_type: Literal['db', 'taylor', 'scm'] = 'db'
    cache_interval: int = 3
    cache_start_step: int = 5
    cache_end_step: Optional[int] = None
    fresh_ratio: float = 0.4

    def to_dict(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'cache_type': self.cache_type,
            'cache_interval': self.cache_interval,
            'cache_start_step': self.cache_start_step,
            'cache_end_step': self.cache_end_step,
            'fresh_ratio': self.fresh_ratio,
        }


@dataclass
class XDiTConfig:
    """Configuration for xDiT/xFuser distributed acceleration.

    xDiT enables distributed inference with sequence and pipeline parallelism
    for faster inference on multi-GPU setups.

    Attributes:
        enabled: Whether to enable xDiT acceleration
        ulysses_degree: Ulysses sequence parallelism degree
        ring_degree: Ring attention parallelism degree
        pipefusion_degree: PipeFusion pipeline parallelism degree
        use_cfg_parallel: Enable CFG (Classifier-Free Guidance) parallelism
        use_torch_compile: Enable torch.compile for additional speedup
        attention_backend: Attention backend to use
    """
    enabled: bool = False
    ulysses_degree: int = 1
    ring_degree: int = 1
    pipefusion_degree: int = 1
    use_cfg_parallel: bool = False
    use_torch_compile: bool = False
    attention_backend: Literal['sdpa', 'flash', 'sage'] = 'flash'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'ulysses_degree': self.ulysses_degree,
            'ring_degree': self.ring_degree,
            'pipefusion_degree': self.pipefusion_degree,
            'use_cfg_parallel': self.use_cfg_parallel,
            'use_torch_compile': self.use_torch_compile,
            'attention_backend': self.attention_backend,
        }


@dataclass
class AccelerationConfig:
    """Combined acceleration configuration.

    Attributes:
        cache_dit: Cache-DiT configuration
        xdit: xDiT configuration
    """
    cache_dit: CacheDiTConfig = field(default_factory=CacheDiTConfig)
    xdit: XDiTConfig = field(default_factory=XDiTConfig)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'AccelerationConfig':
        """Create AccelerationConfig from a dictionary."""
        cache_dit_config = CacheDiTConfig(
            enabled=config.get('enable_cache_dit', False),
            cache_type=config.get('cache_dit_type', 'db'),
            cache_interval=config.get('cache_dit_interval', 3),
            cache_start_step=config.get('cache_dit_start_step', 5),
            cache_end_step=config.get('cache_dit_end_step'),
            fresh_ratio=config.get('cache_dit_fresh_ratio', 0.4),
        )

        xdit_config = XDiTConfig(
            enabled=config.get('enable_xdit', False),
            ulysses_degree=config.get('xdit_ulysses_degree', 1),
            ring_degree=config.get('xdit_ring_degree', 1),
            pipefusion_degree=config.get('xdit_pipefusion_degree', 1),
            use_cfg_parallel=config.get('xdit_use_cfg_parallel', False),
            use_torch_compile=config.get('xdit_use_torch_compile', False),
            attention_backend=config.get('xdit_attention_backend', 'flash'),
        )

        return cls(cache_dit=cache_dit_config, xdit=xdit_config)


def check_cache_dit_available() -> bool:
    """Check if cache-dit is available and compatible."""
    try:
        # Try importing cache_dit - may fail due to diffusers version incompatibility
        import cache_dit
        return True
    except (ImportError, RuntimeError, Exception) as e:
        # RuntimeError can occur if diffusers version is incompatible
        logger.debug(f"cache-dit not available: {e}")
        return False


def check_xdit_available() -> bool:
    """Check if xDiT/xFuser is available."""
    try:
        import xfuser
        return True
    except (ImportError, RuntimeError, Exception) as e:
        logger.debug(f"xfuser not available: {e}")
        return False


def setup_cache_dit(
    model: torch.nn.Module,
    config: CacheDiTConfig,
    pipeline: Optional[Any] = None,
) -> torch.nn.Module:
    """
    Setup cache-dit acceleration for a DiT model.

    This function enables activation caching to speed up inference by
    reusing intermediate computations across diffusion steps.

    Args:
        model: The DiT model to accelerate
        config: Cache-DiT configuration
        pipeline: Optional diffusers pipeline (for easier integration)

    Returns:
        The model with caching enabled

    Example:
        >>> config = CacheDiTConfig(enabled=True, cache_type='db', cache_interval=3)
        >>> model = setup_cache_dit(model, config)
    """
    if not config.enabled:
        logger.info("Cache-DiT is disabled")
        return model

    if not check_cache_dit_available():
        logger.warning(
            "Cache-DiT not available. Install with: pip install cache-dit"
        )
        return model

    try:
        import cache_dit
        from cache_dit import DBCacheConfig, TaylorSeerConfig, SCMConfig

        # Select cache configuration based on type
        if config.cache_type == 'db':
            cache_config = DBCacheConfig(
                cache_interval=config.cache_interval,
                start_step=config.cache_start_step,
                end_step=config.cache_end_step,
            )
        elif config.cache_type == 'taylor':
            cache_config = TaylorSeerConfig(
                fresh_ratio=config.fresh_ratio,
                start_step=config.cache_start_step,
            )
        elif config.cache_type == 'scm':
            cache_config = SCMConfig(
                cache_interval=config.cache_interval,
                start_step=config.cache_start_step,
            )
        else:
            logger.warning(f"Unknown cache type: {config.cache_type}, using DBCache")
            cache_config = DBCacheConfig()

        # Enable caching on the model or pipeline
        if pipeline is not None:
            cache_dit.enable_cache(pipeline, cache_config=cache_config)
            logger.info(f"Cache-DiT enabled on pipeline with {config.cache_type} cache")
        else:
            # For direct model usage, we need to wrap the forward method
            cache_dit.enable_cache(model, cache_config=cache_config)
            logger.info(f"Cache-DiT enabled on model with {config.cache_type} cache")

        return model

    except Exception as e:
        logger.error(f"Failed to setup cache-dit: {e}")
        return model


def setup_xdit(
    config: XDiTConfig,
    world_size: Optional[int] = None,
) -> bool:
    """
    Setup xDiT/xFuser distributed acceleration.

    This function initializes distributed environment for parallel inference.
    Must be called before model initialization in multi-GPU scenarios.

    Args:
        config: xDiT configuration
        world_size: Number of GPUs to use (defaults to all available)

    Returns:
        True if setup was successful, False otherwise

    Example:
        >>> config = XDiTConfig(enabled=True, ulysses_degree=2)
        >>> success = setup_xdit(config)
    """
    if not config.enabled:
        logger.info("xDiT is disabled")
        return False

    if not check_xdit_available():
        logger.warning(
            "xDiT/xFuser not available. Install with: pip install xfuser"
        )
        return False

    try:
        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        import torch.distributed as dist

        if world_size is None:
            world_size = torch.cuda.device_count()

        # Calculate total parallelism
        total_parallel = (
            config.ulysses_degree *
            config.ring_degree *
            config.pipefusion_degree
        )

        if config.use_cfg_parallel:
            total_parallel *= 2

        if total_parallel > world_size:
            logger.warning(
                f"Requested parallelism ({total_parallel}) exceeds available GPUs ({world_size}). "
                f"Adjusting configuration."
            )
            # Adjust to fit available GPUs
            config.ulysses_degree = min(config.ulysses_degree, world_size)
            config.ring_degree = 1
            config.pipefusion_degree = 1

        # Initialize distributed environment if not already done
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')

        init_distributed_environment(
            rank=dist.get_rank(),
            world_size=dist.get_world_size()
        )

        initialize_model_parallel(
            sequence_parallel_degree=config.ulysses_degree * config.ring_degree,
            ring_degree=config.ring_degree,
            ulysses_degree=config.ulysses_degree,
        )

        logger.info(
            f"xDiT initialized with ulysses={config.ulysses_degree}, "
            f"ring={config.ring_degree}, pipefusion={config.pipefusion_degree}"
        )

        return True

    except Exception as e:
        logger.error(f"Failed to setup xDiT: {e}")
        return False


def apply_torch_compile(
    model: torch.nn.Module,
    mode: str = 'reduce-overhead',
) -> torch.nn.Module:
    """
    Apply torch.compile to a model for additional speedup.

    Args:
        model: The model to compile
        mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune')

    Returns:
        The compiled model
    """
    try:
        compiled_model = torch.compile(model, mode=mode)
        logger.info(f"torch.compile applied with mode={mode}")
        return compiled_model
    except Exception as e:
        logger.warning(f"torch.compile failed: {e}")
        return model


class CacheDiTWrapper:
    """
    Wrapper class for managing cache-dit state during sampling.

    This class provides a context manager for enabling/disabling caching
    and handles cache state management across sampling steps.

    Example:
        >>> wrapper = CacheDiTWrapper(model, config)
        >>> with wrapper:
        ...     for step in range(num_steps):
        ...         wrapper.update_step(step)
        ...         output = model(input)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: CacheDiTConfig,
        total_steps: int = 50,
    ):
        self.model = model
        self.config = config
        self.total_steps = total_steps
        self.current_step = 0
        self._cache_enabled = False
        self._original_forward = None

    def __enter__(self):
        if self.config.enabled and check_cache_dit_available():
            self._setup_cache()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup_cache()
        return False

    def _setup_cache(self):
        """Setup caching infrastructure."""
        try:
            import cache_dit
            # Store original forward for restoration
            self._original_forward = self.model.forward
            self._cache_enabled = True
        except Exception as e:
            logger.warning(f"Failed to setup cache: {e}")

    def _cleanup_cache(self):
        """Cleanup caching infrastructure."""
        if self._original_forward is not None:
            self.model.forward = self._original_forward
        self._cache_enabled = False

    def update_step(self, step: int):
        """Update current step for cache management."""
        self.current_step = step

    def should_use_cache(self) -> bool:
        """Check if caching should be used for current step."""
        if not self._cache_enabled:
            return False

        # Check start step
        if self.current_step < self.config.cache_start_step:
            return False

        # Check end step
        if self.config.cache_end_step is not None:
            remaining = self.total_steps - self.current_step
            if remaining <= self.config.cache_end_step:
                return False

        # Check interval
        if self.config.cache_interval > 1:
            if (self.current_step - self.config.cache_start_step) % self.config.cache_interval != 0:
                return True  # Use cached result

        return False  # Compute fresh


def get_acceleration_summary(config: AccelerationConfig) -> str:
    """
    Get a human-readable summary of acceleration configuration.

    Args:
        config: The acceleration configuration

    Returns:
        A formatted string describing the configuration
    """
    lines = ["Acceleration Configuration:"]
    lines.append("=" * 40)

    # Cache-DiT
    if config.cache_dit.enabled:
        lines.append(f"Cache-DiT: ENABLED")
        lines.append(f"  - Type: {config.cache_dit.cache_type}")
        lines.append(f"  - Interval: {config.cache_dit.cache_interval}")
        lines.append(f"  - Start step: {config.cache_dit.cache_start_step}")
        if config.cache_dit.cache_end_step:
            lines.append(f"  - End step: {config.cache_dit.cache_end_step}")
    else:
        lines.append("Cache-DiT: DISABLED")

    lines.append("")

    # xDiT
    if config.xdit.enabled:
        lines.append(f"xDiT: ENABLED")
        lines.append(f"  - Ulysses degree: {config.xdit.ulysses_degree}")
        lines.append(f"  - Ring degree: {config.xdit.ring_degree}")
        lines.append(f"  - PipeFusion degree: {config.xdit.pipefusion_degree}")
        lines.append(f"  - CFG parallel: {config.xdit.use_cfg_parallel}")
        lines.append(f"  - torch.compile: {config.xdit.use_torch_compile}")
    else:
        lines.append("xDiT: DISABLED")

    lines.append("=" * 40)

    return "\n".join(lines)


def estimate_speedup(config: AccelerationConfig) -> Dict[str, float]:
    """
    Estimate expected speedup from acceleration configuration.

    These are rough estimates based on typical benchmarks.
    Actual speedup varies based on model size, hardware, and workload.

    Args:
        config: The acceleration configuration

    Returns:
        Dictionary with estimated speedups for each acceleration method
    """
    speedups = {
        'cache_dit': 1.0,
        'xdit': 1.0,
        'combined': 1.0,
    }

    if config.cache_dit.enabled:
        # Cache-DiT typically provides 1.3-2x speedup
        # More aggressive caching (higher interval) = more speedup but potential quality loss
        cache_speedup = 1.0 + (0.3 * min(config.cache_dit.cache_interval, 5) / 3)
        speedups['cache_dit'] = round(cache_speedup, 2)

    if config.xdit.enabled:
        # xDiT speedup scales with parallelism
        parallel_degree = (
            config.xdit.ulysses_degree *
            config.xdit.ring_degree
        )
        # Assume ~80% efficiency due to communication overhead
        xdit_speedup = parallel_degree * 0.8
        if config.xdit.use_cfg_parallel:
            xdit_speedup *= 1.8  # CFG parallel nearly doubles throughput
        speedups['xdit'] = round(xdit_speedup, 2)

    # Combined speedup (not simply multiplicative due to diminishing returns)
    if config.cache_dit.enabled and config.xdit.enabled:
        speedups['combined'] = round(
            speedups['cache_dit'] * speedups['xdit'] * 0.9, 2
        )
    else:
        speedups['combined'] = max(speedups['cache_dit'], speedups['xdit'])

    return speedups
