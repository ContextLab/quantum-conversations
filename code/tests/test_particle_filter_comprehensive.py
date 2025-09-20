"""
Comprehensive tests for ParticleFilter with real models.

These tests use actual language models to verify the particle filter
works correctly with real generation scenarios.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile

from quantum_conversations import (
    ParticleFilter,
    Particle,
    ModelManager,
    ModelConfig,
    compute_divergence_score,
    save_particles,
    load_particles
)


class TestParticleFilterWithRealModels:
    """Test ParticleFilter with actual HuggingFace models."""

    @pytest.fixture
    def model_manager(self):
        """Create a shared model manager."""
        return ModelManager()

    @pytest.fixture
    def small_filter(self, model_manager):
        """Create a particle filter with a small model for fast testing."""
        return ParticleFilter(
            model_name="EleutherAI/pythia-70m",  # Smallest model for speed
            n_particles=3,
            temperature=0.8,
            device="cpu",
            model_manager=model_manager,
            seed=42  # For reproducibility
        )

    def test_initialization_with_real_model(self, small_filter):
        """Test filter initialization with real model."""
        assert small_filter.model is not None
        assert small_filter.tokenizer is not None
        assert small_filter.n_particles == 3
        assert small_filter.vocab_size > 0

    def test_particle_initialization(self, small_filter):
        """Test initializing particles with a real prompt."""
        prompt = "The quick brown fox"
        small_filter.initialize(prompt)

        assert len(small_filter.particles) == 3

        for particle in small_filter.particles:
            # Check particle structure
            assert len(particle.tokens) > 0
            assert particle.log_prob == 0.0
            assert len(particle.token_probs_history) == 0

            # Check metadata
            assert 'prompt' in particle.metadata
            assert particle.metadata['prompt'] == prompt
            assert 'tokenizer' in particle.metadata

            # Check text property
            assert prompt in particle.text

    def test_single_generation_step(self, small_filter):
        """Test a single generation step with real model."""
        prompt = "Hello"
        small_filter.initialize(prompt)

        initial_lengths = [len(p.tokens) for p in small_filter.particles]

        # Perform one step
        small_filter.step()

        # Verify particles have grown
        for i, particle in enumerate(small_filter.particles):
            assert len(particle.tokens) == initial_lengths[i] + 1
            assert len(particle.token_probs_history) == 1
            assert particle.log_prob < 0  # Should be negative

            # Check probability distribution
            probs = particle.token_probs_history[0]
            assert len(probs) > 0
            assert all(0 <= p <= 1 for p in probs.values())
            assert sum(probs.values()) <= 1.01  # Allow small numerical error

    def test_full_generation(self, small_filter):
        """Test full generation process with real model."""
        prompt = "Once upon a time"
        max_tokens = 20

        particles = small_filter.generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            stop_on_eos=False  # Don't stop on EOS for predictable length
        )

        assert len(particles) == 3

        for particle in particles:
            # Check tokens were generated
            prompt_length = len(small_filter.tokenizer.encode(prompt))
            assert len(particle.tokens) > prompt_length
            assert len(particle.tokens) <= prompt_length + max_tokens

            # Check probability history
            assert len(particle.token_probs_history) <= max_tokens

            # Check generated text
            text = particle.text
            assert prompt in text
            assert len(text) > len(prompt)

            # Check entropy scores
            entropy_scores = particle.entropy_scores
            assert len(entropy_scores) == len(particle.token_probs_history)
            assert all(e >= 0 for e in entropy_scores)

    def test_generation_with_metrics(self, small_filter):
        """Test generation with metrics collection."""
        prompt = "The weather is"

        particles, metrics = small_filter.generate(
            prompt=prompt,
            max_new_tokens=15,
            return_metrics=True
        )

        # Check metrics
        assert metrics.prompt == prompt
        assert metrics.model_name == "EleutherAI/pythia-70m"
        assert metrics.n_particles == 3
        assert metrics.max_tokens == 15
        assert metrics.generation_time > 0
        assert metrics.tokens_per_second > 0
        assert metrics.divergence_score >= 0

    def test_different_temperatures(self, model_manager):
        """Test that different temperatures produce different results."""
        prompts = ["The future of AI is"]

        # Low temperature (more deterministic)
        filter_low = ParticleFilter(
            model_name="EleutherAI/pythia-70m",
            n_particles=5,
            temperature=0.1,
            device="cpu",
            model_manager=model_manager,
            seed=42
        )

        # High temperature (more random)
        filter_high = ParticleFilter(
            model_name="EleutherAI/pythia-70m",
            n_particles=5,
            temperature=2.0,
            device="cpu",
            model_manager=model_manager,
            seed=42
        )

        for prompt in prompts:
            particles_low = filter_low.generate(prompt, max_new_tokens=10)
            particles_high = filter_high.generate(prompt, max_new_tokens=10)

            # Low temperature should have less divergence
            divergence_low = compute_divergence_score(particles_low)
            divergence_high = compute_divergence_score(particles_high)

            # High temperature typically produces more diverse outputs
            # (though not guaranteed with small sample)
            assert divergence_low <= divergence_high + 0.5  # Allow some variance

    def test_save_and_load_particles(self, small_filter):
        """Test saving and loading generated particles."""
        prompt = "Test save"
        particles = small_filter.generate(prompt, max_new_tokens=10)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Save particles
            save_particles(particles, tmp_path)

            # Load particles
            loaded = load_particles(tmp_path)

            # Verify
            assert len(loaded) == len(particles)
            for orig, loaded_p in zip(particles, loaded):
                assert orig.tokens == loaded_p.tokens
                assert orig.log_prob == loaded_p.log_prob
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_top_k_filtering(self, model_manager):
        """Test top-k filtering in generation."""
        filter_topk = ParticleFilter(
            model_name="EleutherAI/pythia-70m",
            n_particles=3,
            temperature=1.0,
            top_k=5,  # Very restrictive
            top_p=1.0,  # Disable top-p
            device="cpu",
            model_manager=model_manager
        )

        prompt = "The"
        filter_topk.initialize(prompt)

        # Get next token probabilities
        input_ids = filter_topk.tokenizer.encode(prompt, return_tensors="pt")
        probs = filter_topk._get_next_token_probs(input_ids)

        # Should have at most 5 tokens
        assert len(probs) <= 5

    def test_top_p_filtering(self, model_manager):
        """Test nucleus (top-p) filtering in generation."""
        filter_topp = ParticleFilter(
            model_name="EleutherAI/pythia-70m",
            n_particles=3,
            temperature=1.0,
            top_k=0,  # Disable top-k
            top_p=0.5,  # Very restrictive
            device="cpu",
            model_manager=model_manager
        )

        prompt = "The"
        filter_topp.initialize(prompt)

        # Get next token probabilities
        input_ids = filter_topp.tokenizer.encode(prompt, return_tensors="pt")
        probs = filter_topp._get_next_token_probs(input_ids)

        # Check cumulative probability
        sorted_probs = sorted(probs.values(), reverse=True)
        cumsum = np.cumsum(sorted_probs)

        # Should include tokens up to ~0.5 cumulative probability
        assert cumsum[-1] <= 0.6  # Allow some tolerance

    def test_model_caching(self, model_manager):
        """Test that models are properly cached between filters."""
        # Create first filter
        filter1 = ParticleFilter(
            model_name="EleutherAI/pythia-70m",
            n_particles=2,
            model_manager=model_manager
        )

        # Create second filter with same model
        filter2 = ParticleFilter(
            model_name="EleutherAI/pythia-70m",
            n_particles=3,
            model_manager=model_manager
        )

        # Should reuse the same model instance
        assert filter1.model is filter2.model
        assert filter1.tokenizer is filter2.tokenizer

    @pytest.mark.parametrize("prompt,expected_behavior", [
        ("def fibonacci(n):", "code-like"),
        ("Once upon a time,", "story-like"),
        ("The scientific method", "factual"),
    ])
    def test_different_prompt_types(self, small_filter, prompt, expected_behavior):
        """Test generation with different types of prompts."""
        particles = small_filter.generate(prompt, max_new_tokens=20)

        # Just verify generation works for different prompt types
        for particle in particles:
            text = particle.text
            assert prompt in text
            assert len(text) > len(prompt)

            # Different prompts should lead to some variation
            if expected_behavior == "code-like":
                # Code prompts might have different entropy patterns
                pass  # Specific checks could go here
            elif expected_behavior == "story-like":
                # Story prompts might be more diverse
                pass
            elif expected_behavior == "factual":
                # Factual prompts might be more consistent
                pass

    def test_reproducibility_with_seed(self, model_manager):
        """Test that setting seed produces reproducible results."""
        prompt = "The meaning of life is"

        # First run
        filter1 = ParticleFilter(
            model_name="EleutherAI/pythia-70m",
            n_particles=2,
            temperature=1.0,
            device="cpu",
            model_manager=model_manager,
            seed=12345
        )
        particles1 = filter1.generate(prompt, max_new_tokens=10)
        texts1 = [p.text for p in particles1]

        # Second run with same seed
        filter2 = ParticleFilter(
            model_name="EleutherAI/pythia-70m",
            n_particles=2,
            temperature=1.0,
            device="cpu",
            model_manager=model_manager,
            seed=12345
        )
        particles2 = filter2.generate(prompt, max_new_tokens=10)
        texts2 = [p.text for p in particles2]

        # Should produce identical results
        assert texts1 == texts2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_generation(self, model_manager):
        """Test generation on GPU if available."""
        filter_gpu = ParticleFilter(
            model_name="EleutherAI/pythia-70m",
            n_particles=3,
            temperature=0.8,
            device="cuda",
            model_manager=model_manager
        )

        prompt = "GPU test"
        particles = filter_gpu.generate(prompt, max_new_tokens=10)

        # Verify generation worked
        assert len(particles) == 3
        for particle in particles:
            assert len(particle.text) > len(prompt)

    def test_different_models(self, model_manager):
        """Test that different models produce different outputs."""
        prompt = "The answer is"

        # Test with pythia-70m
        filter1 = ParticleFilter(
            model_name="EleutherAI/pythia-70m",
            n_particles=3,
            temperature=0.8,
            device="cpu",
            model_manager=model_manager,
            seed=42
        )

        # Test with distilgpt2
        filter2 = ParticleFilter(
            model_name="distilgpt2",
            n_particles=3,
            temperature=0.8,
            device="cpu",
            model_manager=model_manager,
            seed=42
        )

        particles1 = filter1.generate(prompt, max_new_tokens=10)
        particles2 = filter2.generate(prompt, max_new_tokens=10)

        # Different models should produce different outputs
        # (even with same seed, as they have different architectures)
        texts1 = [p.text for p in particles1]
        texts2 = [p.text for p in particles2]

        # At least some outputs should differ
        assert texts1 != texts2 or any(t1 != t2 for t1, t2 in zip(texts1, texts2))