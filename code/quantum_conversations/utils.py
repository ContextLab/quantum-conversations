"""
Utility functions for quantum conversations.

This module provides common utilities for:
- Tensor operations and data structures
- Token sequence analysis
- Probability computations
- Data serialization and loading
- Performance profiling
"""

import torch
import numpy as np
import pickle
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class GenerationMetrics:
    """Metrics for a generation run."""
    prompt: str
    model_name: str
    n_particles: int
    max_tokens: int
    temperature: float
    generation_time: float
    tokens_per_second: float
    memory_usage_mb: Optional[float] = None
    divergence_score: Optional[float] = None
    entropy_scores: Optional[List[float]] = None


@contextmanager
def timer(name: str = "Operation"):
    """Context manager for timing operations."""
    start = time.time()
    logger.info(f"{name} started...")
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.info(f"{name} completed in {elapsed:.2f} seconds")


def get_memory_usage() -> float:
    """Get current GPU memory usage in MB (if available)."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def save_particles(particles: List, filepath: Union[str, Path]) -> None:
    """
    Save particle data to a pickle file.

    Args:
        particles: List of particles to save
        filepath: Path to save file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(particles, f)
    logger.info(f"Saved {len(particles)} particles to {filepath}")


def load_particles(filepath: Union[str, Path]) -> List:
    """
    Load particle data from a pickle file.

    Args:
        filepath: Path to pickle file

    Returns:
        List of particles
    """
    filepath = Path(filepath)
    with open(filepath, 'rb') as f:
        particles = pickle.load(f)
    logger.info(f"Loaded {len(particles)} particles from {filepath}")
    return particles


def save_tensor(tensor: torch.Tensor, filepath: Union[str, Path]) -> None:
    """
    Save a tensor to file.

    Args:
        tensor: Tensor to save
        filepath: Path to save file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, filepath)
    logger.info(f"Saved tensor of shape {tensor.shape} to {filepath}")


def load_tensor(filepath: Union[str, Path]) -> torch.Tensor:
    """
    Load a tensor from file.

    Args:
        filepath: Path to tensor file

    Returns:
        Loaded tensor
    """
    filepath = Path(filepath)
    tensor = torch.load(filepath, map_location='cpu')
    logger.info(f"Loaded tensor of shape {tensor.shape} from {filepath}")
    return tensor


def compute_token_entropy(prob_distribution: Dict[int, float]) -> float:
    """
    Compute entropy of a token probability distribution.

    Args:
        prob_distribution: Dictionary mapping token IDs to probabilities

    Returns:
        Entropy value
    """
    probs = np.array(list(prob_distribution.values()))
    # Avoid log(0)
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy)


def compute_sequence_entropy(token_probs_history: List[Dict[int, float]]) -> List[float]:
    """
    Compute entropy for each step in a sequence.

    Args:
        token_probs_history: List of probability distributions

    Returns:
        List of entropy values
    """
    return [compute_token_entropy(probs) for probs in token_probs_history]


def compute_divergence_score(particles: List) -> float:
    """
    Compute a divergence score measuring how different the particle paths are.

    Args:
        particles: List of particles

    Returns:
        Divergence score (0 = identical, higher = more divergent)
    """
    if len(particles) < 2:
        return 0.0

    # Get token sequences
    sequences = [p.tokens for p in particles]

    # Find minimum common length
    min_length = min(len(seq) for seq in sequences)

    if min_length == 0:
        return 0.0

    # Compute pairwise differences
    total_diff = 0
    comparisons = 0

    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            seq1 = sequences[i][:min_length]
            seq2 = sequences[j][:min_length]

            # Count differences
            diff = sum(t1 != t2 for t1, t2 in zip(seq1, seq2))
            total_diff += diff / min_length
            comparisons += 1

    return total_diff / comparisons if comparisons > 0 else 0.0


def create_probability_tensor(
    particles: List,
    vocab_size: int,
    max_length: Optional[int] = None
) -> torch.Tensor:
    """
    Create a V×T×N tensor from particle data.

    Args:
        particles: List of particles
        vocab_size: Size of vocabulary
        max_length: Maximum sequence length (if None, use longest)

    Returns:
        Tensor of shape (vocab_size, time_steps, n_particles)
    """
    n_particles = len(particles)

    # Determine time steps
    if max_length is None:
        max_length = max(len(p.token_probs_history) for p in particles)

    # Initialize tensor
    tensor = torch.zeros((vocab_size, max_length, n_particles))

    # Fill tensor
    for p_idx, particle in enumerate(particles):
        for t_idx, probs_dict in enumerate(particle.token_probs_history):
            if t_idx >= max_length:
                break
            for token_id, prob in probs_dict.items():
                if token_id < vocab_size:
                    tensor[token_id, t_idx, p_idx] = prob

    return tensor


def aggregate_particle_probabilities(
    particles: List,
    method: str = 'mean'
) -> List[Dict[int, float]]:
    """
    Aggregate probability distributions across particles.

    Args:
        particles: List of particles
        method: Aggregation method ('mean', 'max', 'sum')

    Returns:
        List of aggregated probability distributions
    """
    if not particles or not particles[0].token_probs_history:
        return []

    time_steps = max(len(p.token_probs_history) for p in particles)
    aggregated = []

    for t in range(time_steps):
        # Collect all probabilities at time t
        all_probs = {}
        for particle in particles:
            if t < len(particle.token_probs_history):
                for token_id, prob in particle.token_probs_history[t].items():
                    if token_id not in all_probs:
                        all_probs[token_id] = []
                    all_probs[token_id].append(prob)

        # Aggregate
        step_probs = {}
        for token_id, prob_list in all_probs.items():
            if method == 'mean':
                step_probs[token_id] = np.mean(prob_list)
            elif method == 'max':
                step_probs[token_id] = np.max(prob_list)
            elif method == 'sum':
                step_probs[token_id] = np.sum(prob_list)
            else:
                raise ValueError(f"Unknown aggregation method: {method}")

        aggregated.append(step_probs)

    return aggregated


def get_top_tokens(
    prob_distribution: Dict[int, float],
    tokenizer,
    top_k: int = 10
) -> List[Tuple[str, float]]:
    """
    Get the top-k most probable tokens from a distribution.

    Args:
        prob_distribution: Token probability distribution
        tokenizer: Tokenizer for decoding
        top_k: Number of top tokens to return

    Returns:
        List of (token_text, probability) tuples
    """
    # Sort by probability
    sorted_tokens = sorted(
        prob_distribution.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    # Decode tokens
    result = []
    for token_id, prob in sorted_tokens:
        try:
            token_text = tokenizer.decode([token_id])
            result.append((token_text, prob))
        except:
            result.append((f"<token_{token_id}>", prob))

    return result


def estimate_vocab_size(model_or_tokenizer) -> int:
    """
    Estimate vocabulary size from a model or tokenizer.

    Args:
        model_or_tokenizer: Model or tokenizer object

    Returns:
        Vocabulary size
    """
    # Try tokenizer first
    if hasattr(model_or_tokenizer, 'vocab_size'):
        return model_or_tokenizer.vocab_size

    # Try model config
    if hasattr(model_or_tokenizer, 'config'):
        if hasattr(model_or_tokenizer.config, 'vocab_size'):
            return model_or_tokenizer.config.vocab_size

    # Default fallback
    return 32000


def hash_prompt(prompt: str) -> str:
    """
    Create a hash of a prompt for caching/identification.

    Args:
        prompt: Input prompt

    Returns:
        Hex hash string
    """
    return hashlib.md5(prompt.encode()).hexdigest()[:8]


def format_generation_params(
    model_name: str,
    n_particles: int,
    temperature: float,
    max_tokens: int,
    **kwargs
) -> str:
    """
    Format generation parameters as a readable string.

    Args:
        model_name: Model identifier
        n_particles: Number of particles
        temperature: Temperature setting
        max_tokens: Maximum tokens
        **kwargs: Additional parameters

    Returns:
        Formatted string
    """
    parts = [
        f"Model: {model_name}",
        f"Particles: {n_particles}",
        f"Temperature: {temperature}",
        f"Max tokens: {max_tokens}",
    ]

    for key, value in kwargs.items():
        parts.append(f"{key}: {value}")

    return " | ".join(parts)


def save_metrics(metrics: GenerationMetrics, filepath: Union[str, Path]) -> None:
    """
    Save generation metrics to JSON.

    Args:
        metrics: Generation metrics
        filepath: Path to save file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(metrics)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved metrics to {filepath}")


def load_metrics(filepath: Union[str, Path]) -> GenerationMetrics:
    """
    Load generation metrics from JSON.

    Args:
        filepath: Path to JSON file

    Returns:
        Generation metrics
    """
    filepath = Path(filepath)
    with open(filepath, 'r') as f:
        data = json.load(f)

    return GenerationMetrics(**data)


def batch_generate_prompts(prompts: List[str]) -> List[Dict[str, Any]]:
    """
    Prepare a batch of prompts for parallel generation.

    Args:
        prompts: List of prompts

    Returns:
        List of prompt configurations
    """
    configs = []
    for i, prompt in enumerate(prompts):
        configs.append({
            'id': i,
            'prompt': prompt,
            'hash': hash_prompt(prompt),
            'length': len(prompt)
        })
    return configs