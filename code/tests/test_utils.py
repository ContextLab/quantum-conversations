"""
Unit tests for utility functions using real data.
"""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path
import json

from quantum_conversations.utils import (
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
from quantum_conversations.particle_filter import Particle
from transformers import AutoTokenizer


class TestGenerationMetrics:
    """Test GenerationMetrics dataclass."""

    def test_creation(self):
        """Test creating metrics object."""
        metrics = GenerationMetrics(
            prompt="Test prompt",
            model_name="test-model",
            n_particles=10,
            max_tokens=50,
            temperature=0.8,
            generation_time=2.5,
            tokens_per_second=20.0
        )

        assert metrics.prompt == "Test prompt"
        assert metrics.n_particles == 10
        assert metrics.tokens_per_second == 20.0

    def test_optional_fields(self):
        """Test optional fields in metrics."""
        metrics = GenerationMetrics(
            prompt="Test",
            model_name="model",
            n_particles=5,
            max_tokens=30,
            temperature=1.0,
            generation_time=1.0,
            tokens_per_second=30.0,
            memory_usage_mb=512.0,
            divergence_score=0.75,
            entropy_scores=[1.2, 1.5, 1.8]
        )

        assert metrics.memory_usage_mb == 512.0
        assert metrics.divergence_score == 0.75
        assert len(metrics.entropy_scores) == 3


class TestTimerContext:
    """Test timer context manager."""

    def test_timer_execution(self, caplog):
        """Test that timer logs execution time."""
        import logging
        logging.basicConfig(level=logging.INFO)

        with timer("Test operation"):
            # Simulate some work
            import time
            time.sleep(0.01)

        # Check that timer logged
        assert "Test operation started" in caplog.text or "Test operation completed" in caplog.text


class TestMemoryUtils:
    """Test memory utility functions."""

    def test_get_memory_usage(self):
        """Test getting memory usage."""
        memory = get_memory_usage()
        assert isinstance(memory, float)
        assert memory >= 0


class TestParticleSerialization:
    """Test particle save/load functions."""

    @pytest.fixture
    def sample_particles(self):
        """Create sample particles for testing."""
        particles = []
        for i in range(3):
            particle = Particle(
                tokens=[1, 2, 3, 4, 5],
                log_prob=-2.5 * i,
                token_probs_history=[
                    {10: 0.5, 20: 0.3, 30: 0.2},
                    {15: 0.6, 25: 0.4}
                ],
                metadata={"id": i}
            )
            particles.append(particle)
        return particles

    def test_save_load_particles(self, sample_particles):
        """Test saving and loading particles."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Save particles
            save_particles(sample_particles, tmp_path)
            assert tmp_path.exists()

            # Load particles
            loaded = load_particles(tmp_path)

            # Verify
            assert len(loaded) == len(sample_particles)
            for orig, loaded_p in zip(sample_particles, loaded):
                assert orig.tokens == loaded_p.tokens
                assert orig.log_prob == loaded_p.log_prob
                assert orig.token_probs_history == loaded_p.token_probs_history
        finally:
            tmp_path.unlink(missing_ok=True)


class TestTensorOperations:
    """Test tensor save/load operations."""

    def test_save_load_tensor(self):
        """Test saving and loading tensors."""
        # Create test tensor
        tensor = torch.randn(10, 20, 30)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Save tensor
            save_tensor(tensor, tmp_path)
            assert tmp_path.exists()

            # Load tensor
            loaded = load_tensor(tmp_path)

            # Verify
            assert torch.allclose(tensor, loaded)
            assert tensor.shape == loaded.shape
        finally:
            tmp_path.unlink(missing_ok=True)


class TestEntropyCalculations:
    """Test entropy calculation functions."""

    def test_compute_token_entropy(self):
        """Test computing entropy of token distribution."""
        # Uniform distribution (high entropy)
        uniform_dist = {i: 0.1 for i in range(10)}
        uniform_entropy = compute_token_entropy(uniform_dist)
        assert uniform_entropy > 3.0  # log2(10) ≈ 3.32

        # Peaked distribution (low entropy)
        peaked_dist = {0: 0.9, 1: 0.05, 2: 0.05}
        peaked_entropy = compute_token_entropy(peaked_dist)
        assert peaked_entropy < 1.0

        # Single token (zero entropy)
        single_dist = {0: 1.0}
        single_entropy = compute_token_entropy(single_dist)
        assert single_entropy == 0.0

    def test_compute_sequence_entropy(self):
        """Test computing entropy for sequence of distributions."""
        sequence = [
            {0: 0.5, 1: 0.5},      # High entropy
            {0: 0.9, 1: 0.1},      # Low entropy
            {0: 1.0}               # Zero entropy
        ]

        entropies = compute_sequence_entropy(sequence)

        assert len(entropies) == 3
        assert entropies[0] > entropies[1]
        assert entropies[1] > entropies[2]
        assert entropies[2] == 0.0


class TestDivergenceScore:
    """Test divergence score calculation."""

    def test_identical_particles(self):
        """Test divergence with identical particles."""
        particles = [
            Particle(tokens=[1, 2, 3], log_prob=-1.0, token_probs_history=[], metadata={}),
            Particle(tokens=[1, 2, 3], log_prob=-1.0, token_probs_history=[], metadata={}),
            Particle(tokens=[1, 2, 3], log_prob=-1.0, token_probs_history=[], metadata={})
        ]

        score = compute_divergence_score(particles)
        assert score == 0.0

    def test_different_particles(self):
        """Test divergence with different particles."""
        particles = [
            Particle(tokens=[1, 2, 3], log_prob=-1.0, token_probs_history=[], metadata={}),
            Particle(tokens=[4, 5, 6], log_prob=-1.0, token_probs_history=[], metadata={}),
            Particle(tokens=[7, 8, 9], log_prob=-1.0, token_probs_history=[], metadata={})
        ]

        score = compute_divergence_score(particles)
        assert score == 1.0  # Completely different

    def test_partial_divergence(self):
        """Test partial divergence."""
        particles = [
            Particle(tokens=[1, 2, 3, 4], log_prob=-1.0, token_probs_history=[], metadata={}),
            Particle(tokens=[1, 2, 5, 6], log_prob=-1.0, token_probs_history=[], metadata={}),
            Particle(tokens=[1, 2, 3, 6], log_prob=-1.0, token_probs_history=[], metadata={})
        ]

        score = compute_divergence_score(particles)
        assert 0 < score < 1  # Partial divergence


class TestProbabilityTensor:
    """Test probability tensor creation."""

    def test_create_probability_tensor(self):
        """Test creating V×T×N tensor from particles."""
        particles = []
        vocab_size = 100
        n_particles = 3
        time_steps = 2

        for i in range(n_particles):
            particle = Particle(
                tokens=[1, 2],
                log_prob=-1.0,
                token_probs_history=[
                    {10: 0.5, 20: 0.3, 30: 0.2},
                    {15: 0.6, 25: 0.4}
                ],
                metadata={}
            )
            particles.append(particle)

        tensor = create_probability_tensor(particles, vocab_size)

        # Check shape
        assert tensor.shape == (vocab_size, time_steps, n_particles)

        # Check some values
        assert tensor[10, 0, 0] == 0.5
        assert tensor[15, 1, 0] == 0.6


class TestAggregation:
    """Test probability aggregation functions."""

    def test_aggregate_probabilities_mean(self):
        """Test mean aggregation of probabilities."""
        particles = [
            Particle(
                tokens=[1],
                log_prob=-1.0,
                token_probs_history=[{10: 0.5, 20: 0.3}],
                metadata={}
            ),
            Particle(
                tokens=[2],
                log_prob=-1.0,
                token_probs_history=[{10: 0.7, 20: 0.1}],
                metadata={}
            )
        ]

        aggregated = aggregate_particle_probabilities(particles, method='mean')

        assert len(aggregated) == 1
        assert aggregated[0][10] == pytest.approx(0.6)  # (0.5 + 0.7) / 2
        assert aggregated[0][20] == pytest.approx(0.2)  # (0.3 + 0.1) / 2

    def test_aggregate_probabilities_max(self):
        """Test max aggregation of probabilities."""
        particles = [
            Particle(
                tokens=[1],
                log_prob=-1.0,
                token_probs_history=[{10: 0.5, 20: 0.3}],
                metadata={}
            ),
            Particle(
                tokens=[2],
                log_prob=-1.0,
                token_probs_history=[{10: 0.7, 20: 0.1}],
                metadata={}
            )
        ]

        aggregated = aggregate_particle_probabilities(particles, method='max')

        assert aggregated[0][10] == 0.7  # max(0.5, 0.7)
        assert aggregated[0][20] == 0.3  # max(0.3, 0.1)


class TestTokenUtils:
    """Test token-related utilities."""

    @pytest.fixture
    def tokenizer(self):
        """Get a real tokenizer for testing."""
        return AutoTokenizer.from_pretrained("gpt2")

    def test_get_top_tokens(self, tokenizer):
        """Test getting top tokens from distribution."""
        prob_dist = {
            tokenizer.encode("hello")[0]: 0.5,
            tokenizer.encode("world")[0]: 0.3,
            tokenizer.encode("test")[0]: 0.2
        }

        top_tokens = get_top_tokens(prob_dist, tokenizer, top_k=2)

        assert len(top_tokens) == 2
        assert top_tokens[0][1] == 0.5  # Highest probability
        assert top_tokens[1][1] == 0.3  # Second highest

    def test_estimate_vocab_size(self, tokenizer):
        """Test vocabulary size estimation."""
        vocab_size = estimate_vocab_size(tokenizer)
        assert vocab_size == tokenizer.vocab_size


class TestPromptUtils:
    """Test prompt-related utilities."""

    def test_hash_prompt(self):
        """Test prompt hashing."""
        prompt1 = "Hello world"
        prompt2 = "Hello world"
        prompt3 = "Different prompt"

        hash1 = hash_prompt(prompt1)
        hash2 = hash_prompt(prompt2)
        hash3 = hash_prompt(prompt3)

        # Same prompts should have same hash
        assert hash1 == hash2
        # Different prompts should have different hashes
        assert hash1 != hash3
        # Hash should be 8 characters
        assert len(hash1) == 8

    def test_format_generation_params(self):
        """Test formatting generation parameters."""
        formatted = format_generation_params(
            model_name="test-model",
            n_particles=10,
            temperature=0.8,
            max_tokens=50,
            top_k=40,
            top_p=0.95
        )

        assert "test-model" in formatted
        assert "10" in formatted
        assert "0.8" in formatted
        assert "top_k: 40" in formatted

    def test_batch_generate_prompts(self):
        """Test batch prompt preparation."""
        prompts = ["First prompt", "Second prompt", "Third prompt"]
        configs = batch_generate_prompts(prompts)

        assert len(configs) == 3
        for i, config in enumerate(configs):
            assert config['id'] == i
            assert config['prompt'] == prompts[i]
            assert 'hash' in config
            assert 'length' in config


class TestMetricsSerialization:
    """Test metrics save/load."""

    def test_save_load_metrics(self):
        """Test saving and loading generation metrics."""
        metrics = GenerationMetrics(
            prompt="Test prompt",
            model_name="test-model",
            n_particles=10,
            max_tokens=50,
            temperature=0.8,
            generation_time=2.5,
            tokens_per_second=20.0,
            divergence_score=0.65
        )

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Save metrics
            save_metrics(metrics, tmp_path)
            assert tmp_path.exists()

            # Load metrics
            loaded = load_metrics(tmp_path)

            # Verify
            assert loaded.prompt == metrics.prompt
            assert loaded.n_particles == metrics.n_particles
            assert loaded.divergence_score == metrics.divergence_score
        finally:
            tmp_path.unlink(missing_ok=True)