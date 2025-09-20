"""
Model manager for handling language model operations.

This module provides utilities for:
- Loading ANY open-weights model from HuggingFace
- Accessing model internals for token probability estimation
- Model downloading and caching
- Model configuration and loading
- Memory-efficient model management
- Support for various model architectures (GPT, LLaMA, Mistral, Phi, etc.)

IMPORTANT: This module requires open-weights models where we have access to:
- Model weights for forward passes
- Tokenizers for encoding/decoding
- Logits/probabilities at each generation step
"""

import torch
import gc
import os
import json
from typing import Dict, Optional, List, Tuple, Any
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    BitsAndBytesConfig
)
import logging

logger = logging.getLogger(__name__)


class ModelConfig:
    """Configuration for model loading and inference."""

    # Small models suitable for quick testing (all open-weights)
    SMALL_MODELS = [
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "microsoft/phi-2",
        "EleutherAI/pythia-70m",
        "EleutherAI/pythia-160m",
        "EleutherAI/pythia-410m",
        "gpt2",  # Classic small model
        "distilgpt2",  # Even smaller GPT2
    ]

    # Medium models for balanced performance (all open-weights)
    MEDIUM_MODELS = [
        "mistralai/Mistral-7B-v0.1",
        "meta-llama/Llama-2-7b-hf",
        "EleutherAI/pythia-1.4b",
        "EleutherAI/pythia-2.8b",
        "EleutherAI/gpt-j-6b",
        "facebook/opt-1.3b",
        "facebook/opt-2.7b",
    ]

    # Large models (all open-weights)
    LARGE_MODELS = [
        "meta-llama/Llama-2-13b-hf",
        "meta-llama/Llama-2-70b-hf",
        "EleutherAI/gpt-neox-20b",
        "facebook/opt-6.7b",
        "facebook/opt-13b",
    ]

    # Default model for testing
    DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Tiny model for unit tests (fastest)
    TEST_MODEL = "EleutherAI/pythia-70m"

    @classmethod
    def get_test_config(cls) -> Dict[str, Any]:
        """Get configuration optimized for testing."""
        return {
            "model_name": cls.TEST_MODEL,
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "device_map": None,  # Manual device placement for tests
            "low_cpu_mem_usage": True,
        }

    @classmethod
    def get_production_config(cls, model_name: str) -> Dict[str, Any]:
        """Get configuration optimized for production use."""
        config = {
            "model_name": model_name,
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "low_cpu_mem_usage": True,
        }

        # Use device_map="auto" for GPU if available
        if torch.cuda.is_available():
            config["device_map"] = "auto"

        return config


class ModelManager:
    """
    Manages language model loading, configuration, and lifecycle.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        default_device: Optional[str] = None
    ):
        """
        Initialize the model manager.

        Args:
            cache_dir: Directory for caching models (defaults to HF cache)
            default_device: Default device for models (cuda/cpu/mps)
        """
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface/hub")

        # Determine default device
        if default_device:
            self.default_device = default_device
        elif torch.cuda.is_available():
            self.default_device = "cuda"
        elif torch.backends.mps.is_available():
            self.default_device = "mps"
        else:
            self.default_device = "cpu"

        # Track loaded models
        self.loaded_models: Dict[str, Tuple[PreTrainedModel, PreTrainedTokenizer]] = {}

    def get_device(self, device: Optional[str] = None) -> str:
        """Get the device to use for computation."""
        if device:
            return device
        return self.default_device

    def load_model(
        self,
        model_name: str,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_cache: bool = True
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load a model and tokenizer with specified configuration.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on
            torch_dtype: Data type for model weights
            load_in_8bit: Use 8-bit quantization
            load_in_4bit: Use 4-bit quantization
            use_cache: Whether to use cached model if available

        Returns:
            Tuple of (model, tokenizer)
        """
        # Check cache
        if use_cache and model_name in self.loaded_models:
            logger.info(f"Using cached model: {model_name}")
            return self.loaded_models[model_name]

        device = self.get_device(device)

        # Configure dtype
        if torch_dtype is None:
            if device == "cuda":
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32

        logger.info(f"Loading model {model_name} on {device} with dtype {torch_dtype}")

        # Configure quantization
        quantization_config = None
        if load_in_4bit or load_in_8bit:
            if device != "cuda":
                logger.warning("Quantization only supported on CUDA, disabling")
                load_in_4bit = load_in_8bit = False
            else:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    bnb_4bit_compute_dtype=torch_dtype if load_in_4bit else None
                )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=self.cache_dir
        )

        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "cache_dir": self.cache_dir,
            "low_cpu_mem_usage": True,
        }

        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        if device == "cuda" and not (load_in_4bit or load_in_8bit):
            model_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )

        # Move to device if not using device_map
        if "device_map" not in model_kwargs and not (load_in_4bit or load_in_8bit):
            model = model.to(device)

        # Cache the model
        if use_cache:
            self.loaded_models[model_name] = (model, tokenizer)

        logger.info(f"Successfully loaded {model_name}")
        return model, tokenizer

    def unload_model(self, model_name: str) -> bool:
        """
        Unload a model from memory.

        Args:
            model_name: Model identifier to unload

        Returns:
            True if model was unloaded, False if not found
        """
        if model_name in self.loaded_models:
            model, tokenizer = self.loaded_models[model_name]
            del model
            del tokenizer
            del self.loaded_models[model_name]

            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"Unloaded model: {model_name}")
            return True
        return False

    def unload_all_models(self) -> int:
        """
        Unload all cached models.

        Returns:
            Number of models unloaded
        """
        count = len(self.loaded_models)
        model_names = list(self.loaded_models.keys())

        for model_name in model_names:
            self.unload_model(model_name)

        return count

    def list_cached_models(self) -> List[str]:
        """Get list of currently cached model names."""
        return list(self.loaded_models.keys())

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a model.

        Args:
            model_name: Model identifier

        Returns:
            Dictionary with model information
        """
        info = {
            "name": model_name,
            "loaded": model_name in self.loaded_models,
        }

        if model_name in self.loaded_models:
            model, tokenizer = self.loaded_models[model_name]
            info.update({
                "vocab_size": tokenizer.vocab_size,
                "model_type": model.config.model_type,
                "hidden_size": getattr(model.config, "hidden_size", None),
                "num_layers": getattr(model.config, "num_hidden_layers", None),
                "num_parameters": sum(p.numel() for p in model.parameters()),
                "device": str(next(model.parameters()).device),
                "dtype": str(next(model.parameters()).dtype),
            })

        return info

    def estimate_memory_usage(self, model_name: str, dtype: torch.dtype = torch.float16) -> float:
        """
        Estimate memory usage for a model in GB.

        Args:
            model_name: Model identifier
            dtype: Data type for estimation

        Returns:
            Estimated memory in GB
        """
        # Rough estimates based on common model sizes
        size_map = {
            "pythia-70m": 0.14,
            "pythia-160m": 0.32,
            "pythia-410m": 0.82,
            "pythia-1.4b": 2.8,
            "pythia-2.8b": 5.6,
            "TinyLlama": 2.2,
            "phi-2": 5.4,
            "Mistral-7B": 14.0,
            "Llama-2-7b": 14.0,
        }

        # Check for known model sizes
        for key, size_gb in size_map.items():
            if key.lower() in model_name.lower():
                if dtype == torch.float32:
                    return size_gb * 2  # FP32 uses twice the memory
                elif dtype == torch.int8:
                    return size_gb * 0.5  # INT8 uses half
                return size_gb

        # Default estimate for unknown models
        return 4.0  # Assume ~2B parameter model

    @staticmethod
    def download_model(model_name: str, cache_dir: Optional[str] = None) -> bool:
        """
        Pre-download a model to cache.

        Args:
            model_name: Model identifier to download
            cache_dir: Cache directory

        Returns:
            True if successful
        """
        try:
            logger.info(f"Downloading {model_name}...")

            # Download tokenizer
            AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )

            # Download model (weights only, don't load)
            AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )

            logger.info(f"Successfully downloaded {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            return False

    @staticmethod
    def verify_model_availability(model_name: str) -> bool:
        """
        Check if a model is available on HuggingFace.

        Args:
            model_name: Model identifier to check

        Returns:
            True if model exists and is accessible
        """
        try:
            # Try to load just the config
            from transformers import AutoConfig
            AutoConfig.from_pretrained(model_name)
            return True
        except Exception:
            return False

    @staticmethod
    def get_logits(
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get raw logits from a model for given input.

        Args:
            model: The language model
            input_ids: Input token IDs
            attention_mask: Optional attention mask

        Returns:
            Logits tensor of shape (batch_size, sequence_length, vocab_size)
        """
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            return outputs.logits

    @staticmethod
    def get_next_token_probabilities(
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Get next token probabilities from model.

        Args:
            model: The language model
            input_ids: Input token IDs
            temperature: Temperature for scaling logits
            top_k: Optional top-k filtering
            top_p: Optional nucleus filtering

        Returns:
            Probability distribution over vocabulary
        """
        with torch.no_grad():
            # Get logits
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')

            # Apply top-p filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least one token
                sorted_indices_to_remove[..., :1] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float('Inf')

            # Convert to probabilities
            probs = torch.softmax(logits, dim=-1)
            return probs

    @staticmethod
    def is_open_weights_model(model_name: str) -> bool:
        """
        Check if a model has open weights (not gated or restricted).

        Note: This is a heuristic check. Some models may require authentication
        even if they're technically open-weights.

        Args:
            model_name: Model identifier

        Returns:
            True if model appears to be open-weights
        """
        # Known closed/API-only models (partial list)
        closed_models = [
            "openai/",
            "anthropic/",
            "google/gemini",
            "cohere/",
        ]

        for closed in closed_models:
            if model_name.lower().startswith(closed):
                return False

        # Try to verify by loading config
        return ModelManager.verify_model_availability(model_name)