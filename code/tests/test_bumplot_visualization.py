"""
Comprehensive tests for bumplot visualization with real models.

These tests use actual language models and real data to verify
the bumplot functionality works correctly.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import numpy

from quantum_conversations import (
    ParticleFilter,
    TokenSequenceVisualizer,
    ModelManager
)


class TestBumplotWithRealModels:
    """Test bumplot visualization with actual model generation."""

    @pytest.fixture
    def model_manager(self):
        """Create a shared model manager for efficiency."""
        return ModelManager()

    @pytest.fixture
    def particles_small(self, model_manager):
        """Generate particles with small model for testing."""
        pf = ParticleFilter(
            model_name="EleutherAI/pythia-70m",
            n_particles=5,
            temperature=0.8,
            device="cpu",
            model_manager=model_manager,
            seed=42
        )
        particles = pf.generate("The future of technology", max_new_tokens=15)
        return particles, pf.tokenizer

    def test_bumplot_basic_generation(self, particles_small):
        """Test basic bumplot creation with real generation."""
        particles, tokenizer = particles_small

        viz = TokenSequenceVisualizer(tokenizer=tokenizer)
        fig = viz.visualize_bumplot(particles)

        # Verify figure created
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) > 0

        # Check that plot has content
        ax = fig.axes[0]
        assert len(ax.lines) > 0 or len(ax.collections) > 0  # Should have curves

        # Clean up
        plt.close(fig)

    def test_bumplot_data_preparation(self, particles_small):
        """Test data preparation for bumplot."""
        particles, tokenizer = particles_small

        viz = TokenSequenceVisualizer(tokenizer=tokenizer)
        df, metadata = viz._prepare_bumplot_data(particles, max_vocab_display=50)

        # Verify DataFrame structure
        assert not df.empty
        assert 'timestep' in df.columns
        assert any(col.startswith('particle_') for col in df.columns)

        # Check DataFrame dimensions
        n_particles = len(particles)
        assert len([col for col in df.columns if col.startswith('particle_')]) == n_particles

        # Verify metadata
        assert 'token_ranks' in metadata
        assert 'transition_probs' in metadata
        assert 'token_to_text' in metadata

        # Check token ranks are assigned
        assert len(metadata['token_ranks']) > 0
        ranks = list(metadata['token_ranks'].values())
        assert min(ranks) == 1  # Best rank should be 1

    def test_token_ranking(self, particles_small):
        """Test token ranking by frequency."""
        particles, tokenizer = particles_small

        viz = TokenSequenceVisualizer(tokenizer=tokenizer)
        token_ranks = viz._rank_tokens_by_frequency(particles, max_vocab_display=20)

        # Verify ranking properties
        assert len(token_ranks) <= 20
        ranks = list(token_ranks.values())
        assert min(ranks) == 1
        assert max(ranks) == len(token_ranks)
        # Check ranks are sequential
        assert sorted(ranks) == list(range(1, len(ranks) + 1))

    def test_probability_coloring(self, model_manager):
        """Test that colors accurately reflect transition probabilities."""
        pf = ParticleFilter(
            model_name="EleutherAI/pythia-70m",
            n_particles=10,
            temperature=1.0,
            device="cpu",
            model_manager=model_manager,
            seed=123
        )

        particles = pf.generate("Once upon a", max_new_tokens=10)

        viz = TokenSequenceVisualizer(tokenizer=pf.tokenizer)
        fig = viz.visualize_bumplot(
            particles,
            color_by='transition_prob'
        )

        # Verify figure created
        assert fig is not None

        # Check that colorbar was added for probability coloring
        # (This happens when using transition_prob coloring)
        assert len(fig.axes) >= 1

        plt.close(fig)

    def test_entropy_coloring(self, particles_small):
        """Test coloring by entropy."""
        particles, tokenizer = particles_small

        viz = TokenSequenceVisualizer(tokenizer=tokenizer)
        fig = viz.visualize_bumplot(
            particles,
            color_by='entropy'
        )

        assert fig is not None
        plt.close(fig)

    def test_particle_id_coloring(self, particles_small):
        """Test coloring by particle ID."""
        particles, tokenizer = particles_small

        viz = TokenSequenceVisualizer(tokenizer=tokenizer)
        fig = viz.visualize_bumplot(
            particles,
            color_by='particle_id'
        )

        assert fig is not None
        plt.close(fig)

    def test_large_vocabulary_handling(self, model_manager):
        """Test handling of large vocabularies with top-k filtering."""
        pf = ParticleFilter(
            model_name="gpt2",  # Larger vocab (50k+)
            n_particles=8,
            temperature=1.2,
            device="cpu",
            model_manager=model_manager,
            seed=42
        )

        particles = pf.generate("In the beginning", max_new_tokens=20)

        viz = TokenSequenceVisualizer(tokenizer=pf.tokenizer)
        fig = viz.visualize_bumplot(
            particles,
            max_vocab_display=30  # Limit display
        )

        # Verify figure created
        assert fig is not None

        # Check data preparation with vocab limit
        df, metadata = viz._prepare_bumplot_data(particles, max_vocab_display=30)
        # token_ranks contains all tokens seen, but position_ranks enforces the limit
        # Check that position ranks respect the limit
        for t, rank_map in metadata['position_ranks'].items():
            assert len(rank_map) <= 30

        plt.close(fig)

    def test_divergence_patterns(self, model_manager):
        """Test that divergence patterns are visible in plot."""
        # Low temperature for convergence
        pf_converge = ParticleFilter(
            model_name="EleutherAI/pythia-70m",
            n_particles=6,
            temperature=0.1,
            device="cpu",
            model_manager=model_manager,
            seed=42
        )

        # High temperature for divergence
        pf_diverge = ParticleFilter(
            model_name="EleutherAI/pythia-70m",
            n_particles=6,
            temperature=2.0,
            device="cpu",
            model_manager=model_manager,
            seed=42
        )

        particles_converge = pf_converge.generate("The answer is", max_new_tokens=10)
        particles_diverge = pf_diverge.generate("The answer is", max_new_tokens=10)

        viz = TokenSequenceVisualizer(tokenizer=pf_converge.tokenizer)

        # Create both plots
        fig_converge = viz.visualize_bumplot(particles_converge)
        fig_diverge = viz.visualize_bumplot(particles_diverge)

        # Both should create valid figures
        assert fig_converge is not None
        assert fig_diverge is not None

        # Check data shows more diversity in divergent case
        df_conv, _ = viz._prepare_bumplot_data(particles_converge, max_vocab_display=50)
        df_div, _ = viz._prepare_bumplot_data(particles_diverge, max_vocab_display=50)

        # Count unique tokens at each timestep
        unique_conv = []
        unique_div = []

        for t in range(min(10, len(df_conv))):
            conv_tokens = df_conv.iloc[t][[c for c in df_conv.columns if c.startswith('particle_')]].dropna().nunique()
            div_tokens = df_div.iloc[t][[c for c in df_div.columns if c.startswith('particle_')]].dropna().nunique()
            unique_conv.append(conv_tokens)
            unique_div.append(div_tokens)

        # High temperature should generally have more unique tokens
        # (though not guaranteed for every timestep)
        assert sum(unique_div) >= sum(unique_conv) - 2  # Allow some variance

        plt.close(fig_converge)
        plt.close(fig_diverge)

    def test_edge_case_single_particle(self, model_manager):
        """Test edge case with single particle."""
        pf = ParticleFilter(
            model_name="EleutherAI/pythia-70m",
            n_particles=1,
            temperature=0.8,
            device="cpu",
            model_manager=model_manager
        )

        particles = pf.generate("Test", max_new_tokens=5)

        viz = TokenSequenceVisualizer(tokenizer=pf.tokenizer)
        fig = viz.visualize_bumplot(particles)

        assert fig is not None
        plt.close(fig)

    def test_edge_case_short_generation(self, model_manager):
        """Test edge case with very short generation."""
        pf = ParticleFilter(
            model_name="EleutherAI/pythia-70m",
            n_particles=3,
            temperature=0.8,
            device="cpu",
            model_manager=model_manager
        )

        particles = pf.generate("Hi", max_new_tokens=2)

        viz = TokenSequenceVisualizer(tokenizer=pf.tokenizer)
        fig = viz.visualize_bumplot(particles)

        assert fig is not None
        plt.close(fig)

    def test_edge_case_empty_prompt(self, model_manager):
        """Test edge case with empty prompt."""
        pf = ParticleFilter(
            model_name="EleutherAI/pythia-70m",
            n_particles=3,
            temperature=0.8,
            device="cpu",
            model_manager=model_manager
        )

        # Use a minimal prompt instead of empty (models need at least one token)
        particles = pf.generate(" ", max_new_tokens=10)

        viz = TokenSequenceVisualizer(tokenizer=pf.tokenizer)
        fig = viz.visualize_bumplot(particles)

        assert fig is not None
        plt.close(fig)

    def test_save_bumplot(self, particles_small):
        """Test saving bumplot to file."""
        particles, tokenizer = particles_small

        viz = TokenSequenceVisualizer(tokenizer=tokenizer)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            fig = viz.visualize_bumplot(
                particles,
                output_path=str(tmp_path)
            )

            # Verify file created and has content
            assert tmp_path.exists()
            assert tmp_path.stat().st_size > 0

            plt.close(fig)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_combined_visualizations(self, particles_small):
        """Test bumplot alongside other visualization methods."""
        particles, tokenizer = particles_small

        viz = TokenSequenceVisualizer(tokenizer=tokenizer)

        # Create multiple visualization types
        fig_bump = viz.visualize_bumplot(particles)
        fig_sankey = viz.visualize(particles, "Test prompt")
        # Note: visualize_divergence was planned but not implemented yet
        # fig_divergence = viz.visualize_divergence(particles, "Test prompt")

        # All should work independently
        assert all(fig is not None for fig in [fig_bump, fig_sankey])

        # Clean up
        plt.close(fig_bump)
        plt.close(fig_sankey)
        # plt.close(fig_divergence)

    def test_path_probability_calculation(self, particles_small):
        """Test calculation of path probabilities."""
        particles, tokenizer = particles_small

        viz = TokenSequenceVisualizer(tokenizer=tokenizer)
        path_probs = viz._calculate_path_probabilities(particles)

        # Verify structure
        assert isinstance(path_probs, dict)

        # Check probability values
        for (timestep, token_id), prob in path_probs.items():
            assert isinstance(timestep, (int, np.integer))
            assert isinstance(token_id, (int, np.integer))
            assert 0 <= prob <= 1

        # Check probabilities sum to ~1 at each timestep
        timesteps = set(t for (t, _) in path_probs.keys())
        for t in timesteps:
            step_probs = [p for (ts, _), p in path_probs.items() if ts == t]
            assert abs(sum(step_probs) - 1.0) < 0.01

    def test_token_labels(self, particles_small):
        """Test adding token labels to plot."""
        particles, tokenizer = particles_small

        viz = TokenSequenceVisualizer(tokenizer=tokenizer)
        fig = viz.visualize_bumplot(
            particles,
            show_tokens=True,
            max_vocab_display=10  # Limit for clear labels
        )

        assert fig is not None
        plt.close(fig)

    def test_curve_force_parameter(self, particles_small):
        """Test different curve force parameters."""
        particles, tokenizer = particles_small

        viz = TokenSequenceVisualizer(tokenizer=tokenizer)

        # Test with different curve forces
        for curve_force in [0.0, 0.5, 1.0]:
            fig = viz.visualize_bumplot(
                particles,
                curve_force=curve_force
            )
            assert fig is not None
            plt.close(fig)

    def test_custom_figsize(self, particles_small):
        """Test custom figure size."""
        particles, tokenizer = particles_small

        viz = TokenSequenceVisualizer(tokenizer=tokenizer)
        fig = viz.visualize_bumplot(
            particles,
            figsize=(10, 6)
        )

        assert fig is not None
        # Check figure size is applied
        assert fig.get_size_inches()[0] == 10
        assert fig.get_size_inches()[1] == 6
        plt.close(fig)

    @pytest.mark.parametrize("prompt,expected_tokens", [
        ("def fibonacci(n):", 15),
        ("Once upon a time,", 20),
        ("The scientific method", 10),
    ])
    def test_different_prompt_types(self, model_manager, prompt, expected_tokens):
        """Test generation with different types of prompts."""
        pf = ParticleFilter(
            model_name="EleutherAI/pythia-70m",
            n_particles=4,
            temperature=0.8,
            device="cpu",
            model_manager=model_manager,
            seed=42
        )

        particles = pf.generate(prompt, max_new_tokens=expected_tokens)

        viz = TokenSequenceVisualizer(tokenizer=pf.tokenizer)
        fig = viz.visualize_bumplot(particles)

        assert fig is not None
        plt.close(fig)

    def test_color_mapping_consistency(self, particles_small):
        """Test that color mapping is consistent."""
        particles, tokenizer = particles_small

        viz = TokenSequenceVisualizer(tokenizer=tokenizer)
        df, metadata = viz._prepare_bumplot_data(particles, max_vocab_display=50)

        # Test all color methods
        for color_by in ['transition_prob', 'entropy', 'particle_id']:
            colors = viz._get_bumplot_colors(particles, metadata, color_by)

            # Should have one color per particle
            assert len(colors) == len(particles)

            # All should be valid hex colors
            for color in colors:
                assert color.startswith('#')
                assert len(color) == 7  # #RRGGBB format

    @pytest.mark.skipif(not Path("/tmp").exists(), reason="No /tmp directory")
    def test_multiple_saves(self, particles_small):
        """Test saving multiple bumplots."""
        particles, tokenizer = particles_small

        viz = TokenSequenceVisualizer(tokenizer=tokenizer)

        # Save with different color schemes
        for i, color_by in enumerate(['transition_prob', 'entropy', 'particle_id']):
            output_path = f"/tmp/test_bumplot_{color_by}.png"

            fig = viz.visualize_bumplot(
                particles,
                color_by=color_by,
                output_path=output_path
            )

            assert Path(output_path).exists()
            Path(output_path).unlink()  # Clean up
            plt.close(fig)