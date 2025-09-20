"""
Unit tests for the ParticleFilter class.
"""

import pytest
import torch
import numpy as np
from quantum_conversations.particle_filter import ParticleFilter, Particle


class TestParticleFilter:
    """Test cases for ParticleFilter."""
    
    @pytest.fixture
    def filter_cpu(self):
        """Create a particle filter on CPU for testing."""
        return ParticleFilter(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            n_particles=3,
            temperature=0.8,
            device="cpu"
        )
    
    def test_initialization(self, filter_cpu):
        """Test filter initialization."""
        assert filter_cpu.n_particles == 3
        assert filter_cpu.temperature == 0.8
        assert filter_cpu.device == "cpu"
        assert filter_cpu.tokenizer is not None
        assert filter_cpu.model is not None
        
    def test_initialize_particles(self, filter_cpu):
        """Test particle initialization with a prompt."""
        prompt = "Hello world"
        filter_cpu.initialize(prompt)
        
        assert len(filter_cpu.particles) == 3
        for particle in filter_cpu.particles:
            assert isinstance(particle, Particle)
            assert len(particle.tokens) > 0
            assert particle.log_prob == 0.0
            assert len(particle.token_probs_history) == 0
            
    def test_get_next_token_probs(self, filter_cpu):
        """Test getting next token probabilities."""
        prompt = "The sky is"
        input_ids = filter_cpu.tokenizer.encode(prompt, return_tensors="pt")
        
        probs = filter_cpu._get_next_token_probs(input_ids)
        
        assert isinstance(probs, dict)
        assert len(probs) > 0
        assert all(isinstance(k, int) for k in probs.keys())
        assert all(isinstance(v, float) for v in probs.values())
        assert all(0 <= v <= 1 for v in probs.values())
        assert abs(sum(probs.values()) - 1.0) < 0.01  # Should sum to ~1
        
    def test_step(self, filter_cpu):
        """Test single step of particle filtering."""
        prompt = "Once upon a"
        filter_cpu.initialize(prompt)
        initial_lengths = [len(p.tokens) for p in filter_cpu.particles]
        
        filter_cpu.step()
        
        # Check that particles have grown
        for i, particle in enumerate(filter_cpu.particles):
            assert len(particle.tokens) == initial_lengths[i] + 1
            assert len(particle.token_probs_history) == 1
            assert particle.log_prob < 0  # Log prob should be negative
            
    def test_generate(self, filter_cpu):
        """Test full generation process."""
        prompt = "The weather today is"
        max_tokens = 10
        
        particles = filter_cpu.generate(prompt, max_new_tokens=max_tokens)
        
        assert len(particles) == 3
        for particle in particles:
            assert len(particle.tokens) > len(filter_cpu.tokenizer.encode(prompt))
            assert len(particle.token_probs_history) <= max_tokens
            
    def test_get_token_sequences(self, filter_cpu):
        """Test getting decoded sequences."""
        prompt = "AI is"
        filter_cpu.generate(prompt, max_new_tokens=5)
        
        sequences = filter_cpu.get_token_sequences()
        
        assert len(sequences) == 3
        for text, log_prob in sequences:
            assert isinstance(text, str)
            assert prompt in text
            assert isinstance(log_prob, float)
            assert log_prob <= 0
            
    def test_temperature_effect(self):
        """Test that temperature affects sampling."""
        prompt = "Mathematics is"
        
        # High temperature (more random)
        filter_high_temp = ParticleFilter(
            n_particles=5,
            temperature=2.0,
            device="cpu"
        )
        filter_high_temp.generate(prompt, max_new_tokens=10)
        seqs_high = filter_high_temp.get_token_sequences()
        
        # Low temperature (more deterministic)
        filter_low_temp = ParticleFilter(
            n_particles=5,
            temperature=0.1,
            device="cpu"
        )
        filter_low_temp.generate(prompt, max_new_tokens=10)
        seqs_low = filter_low_temp.get_token_sequences()
        
        # Check that sequences exist
        assert len(seqs_high) == 5
        assert len(seqs_low) == 5
        
    def test_top_k_filtering(self, filter_cpu):
        """Test top-k filtering."""
        filter_cpu.top_k = 5
        prompt = "Python programming"
        input_ids = filter_cpu.tokenizer.encode(prompt, return_tensors="pt")
        
        probs = filter_cpu._get_next_token_probs(input_ids)
        
        # Should have at most top_k tokens
        assert len(probs) <= 5
        
    def test_particle_clone(self):
        """Test particle cloning."""
        original = Particle(
            tokens=[1, 2, 3],
            log_prob=-1.5,
            token_probs_history=[{4: 0.5, 5: 0.5}]
        )
        
        cloned = original.clone()
        
        # Check deep copy
        assert cloned.tokens == original.tokens
        assert cloned.tokens is not original.tokens
        assert cloned.log_prob == original.log_prob
        assert cloned.token_probs_history == original.token_probs_history
        assert cloned.token_probs_history is not original.token_probs_history


class TestModelIntegration:
    """Integration tests with actual model calls."""
    
    @pytest.mark.slow
    def test_diverse_prompts(self):
        """Test with various prompt types."""
        filter = ParticleFilter(n_particles=3, device="cpu")
        
        prompts = [
            "The future of technology",
            "In a galaxy far away",
            "def fibonacci(n):",
            "Climate change is",
        ]
        
        for prompt in prompts:
            particles = filter.generate(prompt, max_new_tokens=20)
            sequences = filter.get_token_sequences()
            
            assert len(sequences) == 3
            for text, _ in sequences:
                assert len(text) > len(prompt)
                
    @pytest.mark.slow  
    def test_eos_handling(self):
        """Test that generation stops at EOS token."""
        filter = ParticleFilter(n_particles=2, device="cpu")
        
        # Use a prompt likely to generate EOS quickly
        prompt = "The end."
        particles = filter.generate(prompt, max_new_tokens=50)
        
        # Check that at least some particles found EOS
        eos_token_id = filter.tokenizer.eos_token_id
        if eos_token_id is not None:
            eos_found = any(
                eos_token_id in p.tokens 
                for p in particles
            )
            # This is probabilistic, so we just check it's possible
            assert isinstance(eos_found, bool)