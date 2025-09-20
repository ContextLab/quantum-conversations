"""
Unit tests for the TokenSequenceVisualizer class.
"""

import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import numpy as np
from quantum_conversations.particle_filter import ParticleFilter, Particle
from quantum_conversations.visualizer import TokenSequenceVisualizer
from transformers import AutoTokenizer


class TestTokenSequenceVisualizer:
    """Test cases for TokenSequenceVisualizer."""
    
    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer for testing."""
        return AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    @pytest.fixture
    def visualizer(self, tokenizer):
        """Create a visualizer for testing."""
        return TokenSequenceVisualizer(
            tokenizer=tokenizer,
            figsize=(12, 8),
            alpha=0.4
        )
        
    @pytest.fixture
    def sample_particles(self, tokenizer):
        """Create sample particles for testing."""
        # Create mock particles with known token sequences
        particles = []
        
        # Particle 1: "Hello world today"
        tokens1 = tokenizer.encode("Hello world today")
        particle1 = Particle(
            tokens=tokens1,
            log_prob=-5.2,
            token_probs_history=[
                {tokens1[i]: 0.8, tokens1[i]+1: 0.2} 
                for i in range(1, len(tokens1))
            ]
        )
        particles.append(particle1)
        
        # Particle 2: "Hello there friend"  
        tokens2 = tokenizer.encode("Hello there friend")
        particle2 = Particle(
            tokens=tokens2,
            log_prob=-6.1,
            token_probs_history=[
                {tokens2[i]: 0.7, tokens2[i]+1: 0.3}
                for i in range(1, len(tokens2))
            ]
        )
        particles.append(particle2)
        
        return particles
        
    def test_initialization(self, visualizer, tokenizer):
        """Test visualizer initialization."""
        assert visualizer.tokenizer == tokenizer
        assert visualizer.figsize == (12, 8)
        assert visualizer.alpha == 0.4
        assert visualizer.max_tokens_display == 20
        
    def test_get_token_positions(self, visualizer, sample_particles):
        """Test token position calculation."""
        positions = visualizer._get_token_positions(sample_particles, 0)
        
        assert isinstance(positions, dict)
        # Both particles share the same first token
        assert len(positions) == 1
        assert all(isinstance(k, int) for k in positions.keys())
        assert all(isinstance(v, float) for v in positions.values())
        
    def test_visualize_basic(self, visualizer, sample_particles, tmp_path):
        """Test basic visualization."""
        output_path = tmp_path / "test_sankey.png"
        
        fig = visualizer.visualize(
            particles=sample_particles,
            prompt="Hello",
            output_path=str(output_path)
        )
        
        assert isinstance(fig, plt.Figure)
        assert output_path.exists()
        plt.close(fig)
        
    def test_visualize_with_title(self, visualizer, sample_particles):
        """Test visualization with custom title."""
        fig = visualizer.visualize(
            particles=sample_particles,
            prompt="Test",
            title="Custom Test Title"
        )
        
        assert fig.axes[0].get_title() == "Custom Test Title"
        plt.close(fig)
        
    def test_probability_heatmap(self, visualizer, sample_particles, tmp_path):
        """Test probability heatmap visualization."""
        output_path = tmp_path / "test_heatmap.png"
        
        fig = visualizer.visualize_probability_heatmap(
            particles=sample_particles,
            prompt="Hello",
            top_k_tokens=5,
            output_path=str(output_path)
        )
        
        assert isinstance(fig, plt.Figure)
        assert output_path.exists()
        plt.close(fig)
        
    def test_empty_particles(self, visualizer):
        """Test handling of empty particle list."""
        fig = visualizer.visualize(
            particles=[],
            prompt="Empty test"
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
    def test_long_sequences(self, visualizer, tokenizer):
        """Test handling of sequences longer than max_tokens_display."""
        # Create a particle with many tokens
        long_text = " ".join(["word"] * 50)
        tokens = tokenizer.encode(long_text)
        
        particle = Particle(
            tokens=tokens,
            log_prob=-50.0,
            token_probs_history=[{t: 0.5} for t in tokens[1:]]
        )
        
        fig = visualizer.visualize(
            particles=[particle],
            prompt="Long sequence"
        )
        
        # Should truncate to max_tokens_display
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        

class TestVisualizerIntegration:
    """Integration tests with ParticleFilter."""
    
    @pytest.mark.slow
    def test_full_pipeline(self, tmp_path):
        """Test full pipeline from generation to visualization."""
        # Generate particles
        filter = ParticleFilter(
            n_particles=3,
            temperature=0.8,
            device="cpu"
        )
        
        prompt = "The meaning of life is"
        particles = filter.generate(prompt, max_new_tokens=15)
        
        # Visualize
        visualizer = TokenSequenceVisualizer(
            tokenizer=filter.tokenizer,
            figsize=(15, 10)
        )
        
        # Test Sankey diagram
        sankey_path = tmp_path / "sankey.png"
        fig1 = visualizer.visualize(
            particles=particles,
            prompt=prompt,
            output_path=str(sankey_path)
        )
        assert sankey_path.exists()
        plt.close(fig1)
        
        # Test heatmap
        heatmap_path = tmp_path / "heatmap.png"
        fig2 = visualizer.visualize_probability_heatmap(
            particles=particles,
            prompt=prompt,
            output_path=str(heatmap_path)
        )
        assert heatmap_path.exists()
        plt.close(fig2)
        
    @pytest.mark.slow
    def test_diverse_particles(self):
        """Test visualization with diverse particle paths."""
        filter = ParticleFilter(
            n_particles=5,
            temperature=1.5,  # High temperature for diversity
            device="cpu"
        )
        
        prompt = "In the beginning"
        particles = filter.generate(prompt, max_new_tokens=10)
        
        visualizer = TokenSequenceVisualizer(
            tokenizer=filter.tokenizer,
            alpha=0.3
        )
        
        fig = visualizer.visualize(particles=particles, prompt=prompt)
        
        # Check that figure was created successfully
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) > 0
        plt.close(fig)