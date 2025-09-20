"""
Quantum Conversations: Visualizing token generation paths in language models.

A comprehensive toolkit for exploring multiple token generation paths in
language models using particle filters. Supports any open-weights model
from HuggingFace.
"""

from .particle_filter import ParticleFilter, Particle
from .visualizer import TokenSequenceVisualizer
from .model_manager import ModelManager, ModelConfig
from .utils import (
    GenerationMetrics,
    timer,
    get_memory_usage,
    save_particles,
    load_particles,
    save_tensor,
    load_tensor,
    compute_token_entropy,
    compute_sequence_entropy,
    compute_divergence_score,
    create_probability_tensor,
    aggregate_particle_probabilities,
    get_top_tokens,
    estimate_vocab_size,
    hash_prompt,
    format_generation_params,
    save_metrics,
    load_metrics,
    batch_generate_prompts
)

__version__ = "0.3.0"  # Updated with bumplot visualization
__all__ = [
    # Core classes
    "ParticleFilter",
    "Particle",
    "TokenSequenceVisualizer",
    "ModelManager",
    "ModelConfig",
    "GenerationMetrics",

    # Utility functions
    "timer",
    "get_memory_usage",
    "save_particles",
    "load_particles",
    "save_tensor",
    "load_tensor",
    "compute_token_entropy",
    "compute_sequence_entropy",
    "compute_divergence_score",
    "create_probability_tensor",
    "aggregate_particle_probabilities",
    "get_top_tokens",
    "estimate_vocab_size",
    "hash_prompt",
    "format_generation_params",
    "save_metrics",
    "load_metrics",
    "batch_generate_prompts"
]