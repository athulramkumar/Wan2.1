"""
Model Manager - Loads and holds Wan2.1 models in memory.

Loads both 14B and 1.3B models at startup and keeps them ready for inference.
"""

import logging
import sys
import os
from typing import Optional

import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wan.text2video import WanT2V
from wan.configs.wan_t2v_14B import t2v_14B
from wan.configs.wan_t2v_1_3B import t2v_1_3B

from .config import CHECKPOINT_DIR_14B, CHECKPOINT_DIR_1_3B

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages Wan2.1 models - loads at startup and keeps in memory.
    
    Provides:
    - Model loading with proper GPU placement
    - Access to individual models or both
    - GPU memory monitoring
    - Thread-safe model access (single worker, so no locking needed)
    """
    
    def __init__(self, device_id: int = 0):
        """
        Initialize the model manager.
        
        Args:
            device_id: GPU device ID to use
        """
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}")
        
        # Models (None until loaded)
        self.model_14B: Optional[WanT2V] = None
        self.model_1_3B: Optional[WanT2V] = None
        
        # Loading state
        self._is_loaded = False
        self._loading_error: Optional[str] = None
    
    @property
    def is_loaded(self) -> bool:
        """Check if all models are loaded."""
        return self._is_loaded
    
    @property
    def loaded_models(self) -> list[str]:
        """Get list of loaded model names."""
        models = []
        if self.model_14B is not None:
            models.append("14B")
        if self.model_1_3B is not None:
            models.append("1.3B")
        return models
    
    def get_gpu_memory(self) -> dict:
        """Get current GPU memory usage."""
        if torch.cuda.is_available():
            return {
                "allocated_gb": round(torch.cuda.memory_allocated(self.device_id) / 1e9, 2),
                "reserved_gb": round(torch.cuda.memory_reserved(self.device_id) / 1e9, 2),
                "max_allocated_gb": round(torch.cuda.max_memory_allocated(self.device_id) / 1e9, 2),
            }
        return {"allocated_gb": 0, "reserved_gb": 0, "max_allocated_gb": 0}
    
    def load_models(self) -> bool:
        """
        Load both 14B and 1.3B models.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("=" * 60)
            logger.info("LOADING MODELS")
            logger.info("=" * 60)
            
            # Load 14B model
            logger.info(f"Loading 14B model from {CHECKPOINT_DIR_14B}...")
            self.model_14B = WanT2V(
                config=t2v_14B,
                checkpoint_dir=CHECKPOINT_DIR_14B,
                device_id=self.device_id,
                rank=0,
                t5_fsdp=False,
                dit_fsdp=False,
                use_usp=False,
                t5_cpu=False,
            )
            mem = self.get_gpu_memory()
            logger.info(f"✓ 14B model loaded. GPU memory: {mem['allocated_gb']:.2f} GB")
            
            # Load 1.3B model
            logger.info(f"Loading 1.3B model from {CHECKPOINT_DIR_1_3B}...")
            self.model_1_3B = WanT2V(
                config=t2v_1_3B,
                checkpoint_dir=CHECKPOINT_DIR_1_3B,
                device_id=self.device_id,
                rank=0,
                t5_fsdp=False,
                dit_fsdp=False,
                use_usp=False,
                t5_cpu=False,
            )
            mem = self.get_gpu_memory()
            logger.info(f"✓ 1.3B model loaded. GPU memory: {mem['allocated_gb']:.2f} GB")
            
            self._is_loaded = True
            logger.info("=" * 60)
            logger.info("✓ ALL MODELS LOADED SUCCESSFULLY")
            logger.info(f"  Total GPU memory: {mem['allocated_gb']:.2f} GB")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            self._loading_error = str(e)
            logger.error(f"Failed to load models: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_model(self, model_name: str) -> WanT2V:
        """
        Get a specific model by name.
        
        Args:
            model_name: "14B" or "1.3B"
        
        Returns:
            The requested model
        
        Raises:
            ValueError: If model name is invalid
            RuntimeError: If models not loaded
        """
        if not self._is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        if model_name == "14B":
            return self.model_14B
        elif model_name == "1.3B":
            return self.model_1_3B
        else:
            raise ValueError(f"Unknown model: {model_name}. Must be '14B' or '1.3B'")
    
    def get_models(self) -> dict[str, WanT2V]:
        """
        Get dictionary of all loaded models.
        
        Returns:
            Dict mapping model names to model instances
        """
        if not self._is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        return {
            "14B": self.model_14B,
            "1.3B": self.model_1_3B,
        }
    
    def cleanup(self):
        """Release model resources."""
        logger.info("Cleaning up models...")
        
        if self.model_14B is not None:
            del self.model_14B
            self.model_14B = None
        
        if self.model_1_3B is not None:
            del self.model_1_3B
            self.model_1_3B = None
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._is_loaded = False
        logger.info("✓ Models cleaned up")


# Global model manager instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


def initialize_models(device_id: int = 0) -> ModelManager:
    """
    Initialize and load all models.
    
    This should be called once at server startup.
    
    Args:
        device_id: GPU device ID
    
    Returns:
        The initialized ModelManager
    """
    global _model_manager
    _model_manager = ModelManager(device_id=device_id)
    _model_manager.load_models()
    return _model_manager

