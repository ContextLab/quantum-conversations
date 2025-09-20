"""
Particle filter implementation for tracking token generation paths.

This module implements a particle filter approach for exploring multiple
possible token generation paths in language models. It supports any
open-weights model from HuggingFace.
"""

import numpy as np
import torch
import time
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

from .model_manager import ModelManager
from .utils import (
    timer,
    get_memory_usage,
    compute_token_entropy,
    compute_divergence_score,
    save_particles,
    GenerationMetrics
)

logger = logging.getLogger(__name__)


@dataclass
class Particle:
    """Represents a single particle in the filter."""
    tokens: List[int]
    log_prob: float
    token_probs_history: List[Dict[int, float]]  # List of {token_id: prob} for each step
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata

    def clone(self):
        """Create a deep copy of the particle."""
        return Particle(
            tokens=self.tokens.copy(),
            log_prob=self.log_prob,
            token_probs_history=[h.copy() for h in self.token_probs_history],
            metadata=self.metadata.copy()
        )

    @property
    def text(self) -> str:
        """Get decoded text (requires tokenizer to be set in metadata)."""
        if 'tokenizer' in self.metadata:
            return self.metadata['tokenizer'].decode(self.tokens, skip_special_tokens=True)
        return f"<{len(self.tokens)} tokens>"

    @property
    def entropy_scores(self) -> List[float]:
        """Get entropy scores for each step."""
        return [compute_token_entropy(probs) for probs in self.token_probs_history]


class ParticleFilter:
    """
    Particle filter for tracking multiple hypotheses in language generation.

    Supports any open-weights model from HuggingFace and provides detailed
    tracking of token probabilities across multiple generation paths.
    """

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        n_particles: int = 10,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        device: Optional[str] = None,
        model_manager: Optional[ModelManager] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the particle filter.

        Args:
            model_name: Any HuggingFace open-weights model identifier
            n_particles: Number of particles to maintain
            temperature: Sampling temperature
            top_k: Top-k filtering parameter
            top_p: Top-p (nucleus) filtering parameter
            device: Device to run on (cuda/cpu/mps)
            model_manager: Optional ModelManager instance for model loading
            seed: Random seed for reproducibility
        """
        self.model_name = model_name
        self.n_particles = n_particles
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        # Use provided model manager or create new one
        self.model_manager = model_manager or ModelManager()

        # Load model and tokenizer
        logger.info(f"Loading model: {model_name}")
        self.model, self.tokenizer = self.model_manager.load_model(
            model_name=model_name,
            device=device,
            use_cache=True
        )

        self.device = str(next(self.model.parameters()).device)
        self.vocab_size = self.tokenizer.vocab_size

        self.particles: List[Particle] = []
        self.generation_metrics: Optional[GenerationMetrics] = None
        
    def initialize(self, prompt: str) -> None:
        """
        Initialize particles with a prompt.

        Args:
            prompt: Initial text prompt
        """
        # Tokenize the prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        if hasattr(input_ids, 'to'):
            input_ids = input_ids.to(self.device)

        # Initialize all particles with the same prompt
        self.particles = []
        for i in range(self.n_particles):
            particle = Particle(
                tokens=input_ids[0].cpu().tolist(),
                log_prob=0.0,
                token_probs_history=[],
                metadata={
                    'particle_id': i,
                    'prompt': prompt,
                    'tokenizer': self.tokenizer
                }
            )
            self.particles.append(particle)

        logger.info(f"Initialized {self.n_particles} particles with prompt: '{prompt[:50]}...'")
            
    def _get_next_token_probs(self, input_ids: torch.Tensor) -> Dict[int, float]:
        """
        Get next token probabilities from the model.

        Works with any open-weights HuggingFace model by accessing logits directly.

        Args:
            input_ids: Input token IDs

        Returns:
            Dictionary mapping token IDs to probabilities
        """
        # Use model manager's static method for consistency
        probs_tensor = ModelManager.get_next_token_probabilities(
            model=self.model,
            input_ids=input_ids,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p
        )

        # Convert to dictionary format
        probs = probs_tensor[0] if probs_tensor.dim() > 1 else probs_tensor

        # Get non-zero probability tokens
        non_zero_mask = probs > 1e-6
        token_ids = torch.where(non_zero_mask)[0]
        token_probs = probs[non_zero_mask]

        return {
            int(token_id): float(prob)
            for token_id, prob in zip(token_ids, token_probs)
        }
            
    def step(self, max_new_tokens: int = 1) -> None:
        """
        Perform one step of particle filtering.
        
        Args:
            max_new_tokens: Number of tokens to generate in this step
        """
        new_particles = []
        
        for particle in self.particles:
            # Get current token sequence
            input_ids = torch.tensor([particle.tokens], device=self.device)
            
            # Get next token probabilities
            token_probs = self._get_next_token_probs(input_ids)
            
            # Store probability distribution for visualization
            particle.token_probs_history.append(token_probs)
            
            # Sample next token
            tokens = list(token_probs.keys())
            probs = np.array(list(token_probs.values()))
            
            if tokens:  # Only sample if we have valid tokens
                # Normalize probabilities to ensure they sum to 1
                probs = probs / probs.sum()
                sampled_token = np.random.choice(tokens, p=probs)
                sampled_prob = token_probs[sampled_token]
                
                # Create new particle
                new_particle = particle.clone()
                new_particle.tokens.append(sampled_token)
                new_particle.log_prob += np.log(sampled_prob + 1e-10)
                new_particles.append(new_particle)
            else:
                # If no valid tokens, keep the particle as is
                new_particles.append(particle)
                
        self.particles = new_particles
        
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        stop_on_eos: bool = True,
        return_metrics: bool = False,
        save_path: Optional[Union[str, Path]] = None
    ) -> Union[List[Particle], Tuple[List[Particle], GenerationMetrics]]:
        """
        Generate token sequences using the particle filter.

        Args:
            prompt: Initial text prompt
            max_new_tokens: Maximum number of tokens to generate
            stop_on_eos: Whether to stop when all particles hit EOS
            return_metrics: Whether to return generation metrics
            save_path: Optional path to save particles

        Returns:
            List of final particles, optionally with metrics
        """
        start_time = time.time()
        initial_memory = get_memory_usage()

        with timer(f"Generating {max_new_tokens} tokens with {self.n_particles} particles"):
            self.initialize(prompt)

            for step_num in range(max_new_tokens):
                self.step()

                # Log progress periodically
                if (step_num + 1) % 10 == 0:
                    logger.info(f"Generated {step_num + 1}/{max_new_tokens} tokens")

                # Check if all particles have hit EOS
                if stop_on_eos and self.tokenizer.eos_token_id is not None:
                    all_done = all(
                        self.tokenizer.eos_token_id in p.tokens
                        for p in self.particles
                    )
                    if all_done:
                        logger.info(f"All particles reached EOS at step {step_num + 1}")
                        break

        # Compute metrics
        generation_time = time.time() - start_time
        total_tokens = sum(len(p.tokens) for p in self.particles)
        tokens_per_second = total_tokens / generation_time if generation_time > 0 else 0

        self.generation_metrics = GenerationMetrics(
            prompt=prompt,
            model_name=self.model_name,
            n_particles=self.n_particles,
            max_tokens=max_new_tokens,
            temperature=self.temperature,
            generation_time=generation_time,
            tokens_per_second=tokens_per_second,
            memory_usage_mb=get_memory_usage() - initial_memory,
            divergence_score=compute_divergence_score(self.particles)
        )

        # Save if requested
        if save_path:
            save_particles(self.particles, save_path)

        if return_metrics:
            return self.particles, self.generation_metrics
        return self.particles
    
    def get_token_sequences(self) -> List[Tuple[str, float]]:
        """
        Get decoded token sequences from all particles.
        
        Returns:
            List of (text, log_probability) tuples
        """
        sequences = []
        for particle in self.particles:
            text = self.tokenizer.decode(particle.tokens, skip_special_tokens=True)
            sequences.append((text, particle.log_prob))
        return sequences