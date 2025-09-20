"""
Visualization tools for token sequences and particle paths.

Provides multiple visualization types:
- Sankey-like diagrams for token generation paths
- Probability heatmaps
- Divergence plots
- Token distribution visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
import colorsys
from pathlib import Path as FilePath
import logging
from bumplot import bumplot

from .particle_filter import Particle
from .utils import (
    compute_token_entropy,
    compute_sequence_entropy,
    compute_divergence_score,
    aggregate_particle_probabilities,
    get_top_tokens
)

logger = logging.getLogger(__name__)

# Set default plot style
plt.style.use('seaborn-v0_8-darkgrid') if 'seaborn-v0_8-darkgrid' in plt.style.available else plt.style.use('seaborn-darkgrid' if 'seaborn-darkgrid' in plt.style.available else 'default')
sns.set_palette("husl")


class TokenSequenceVisualizer:
    """
    Comprehensive visualization suite for token sequences and particle paths.

    Supports multiple visualization types for analyzing language model
    generation patterns from particle filter outputs.
    """
    
    def __init__(
        self,
        tokenizer,
        figsize: Tuple[int, int] = (20, 12),
        max_tokens_display: int = 50,
        alpha: float = 0.01,
        line_width: float = 0.5,
        style: str = 'default',
        save_dpi: int = 300
    ):
        """
        Initialize the visualizer.

        Args:
            tokenizer: HuggingFace tokenizer for decoding tokens
            figsize: Figure size for the plot
            max_tokens_display: Maximum number of tokens to display
            alpha: Transparency for particle paths
            line_width: Width of the path lines
            style: Plotting style ('default', 'dark', 'minimal')
            save_dpi: DPI for saved figures
        """
        self.tokenizer = tokenizer
        self.figsize = figsize
        self.max_tokens_display = max_tokens_display
        self.alpha = alpha
        self.line_width = line_width
        self.style = style
        self.save_dpi = save_dpi
        self.vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 32000
        
    def visualize(
        self,
        particles: List[Particle],
        prompt: str,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        highlight_most_probable: bool = True
    ) -> plt.Figure:
        """
        Create a Sankey-like visualization of token sequences.
        
        Args:
            particles: List of particles from the filter
            prompt: Initial prompt
            output_path: Path to save the figure
            title: Plot title
            highlight_most_probable: Whether to highlight the most probable path
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if title is None:
            title = f"Token Generation Paths: \"{prompt}\""
        
        # Find the maximum sequence length
        max_length = min(
            max(len(p.tokens) for p in particles),
            self.max_tokens_display
        )
        
        # Find the most probable particle
        most_probable_idx = None
        if highlight_most_probable and particles:
            log_probs = [p.log_prob for p in particles]
            most_probable_idx = np.argmax(log_probs)
        
        # Draw paths for each particle
        for i, particle in enumerate(particles):
            # Create path coordinates
            x_coords = []
            y_coords = []
            
            for t in range(min(len(particle.tokens), max_length)):
                x_coords.append(t)
                # Map token to y-position (spread tokens vertically)
                # Use token ID modulo to distribute tokens
                y_pos = (particle.tokens[t] % 1000) / 10.0 - 50.0
                y_coords.append(y_pos)
            
            # Determine if this is the most probable path
            is_most_probable = (i == most_probable_idx)
            
            # Set line properties
            if is_most_probable:
                color = 'red'
                alpha = 1.0
                linewidth = 2.0
                zorder = 1000  # Draw on top
            else:
                color = 'black'
                alpha = self.alpha
                linewidth = self.line_width
                zorder = 1
            
            # Draw the path
            if len(x_coords) > 1:
                ax.plot(x_coords, y_coords, 
                       color=color, alpha=alpha, linewidth=linewidth,
                       zorder=zorder)
        
        # Add token labels for the most probable sequence
        if highlight_most_probable and most_probable_idx is not None:
            most_probable_particle = particles[most_probable_idx]
            
            # Add text below the plot showing the most probable sequence
            text = self.tokenizer.decode(most_probable_particle.tokens[:max_length])
            ax.text(0.5, -0.15, f"Most probable sequence: {text}",
                   transform=ax.transAxes,
                   ha='center', va='top',
                   fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
        
        # Set axis properties
        ax.set_xlim(-0.5, max_length - 0.5)
        ax.set_ylim(-60, 60)
        ax.set_xlabel("Token Position", fontsize=12)
        ax.set_ylabel("Token Space", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Remove y-axis ticks (too many tokens to label)
        ax.set_yticks([])
        
        # Add grid
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add legend
        ax.text(
            0.02, 0.98,
            f"Particles: {len(particles)}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.save_dpi, bbox_inches='tight')

        return fig
    
    def visualize_probability_heatmap(
        self,
        particles: List[Particle],
        prompt: str,
        output_path: Optional[str] = None,
        vocab_size: Optional[int] = None
    ) -> plt.Figure:
        """
        Create a heatmap showing all token probabilities over time.
        
        Args:
            particles: List of particles from the filter
            prompt: Initial prompt
            output_path: Path to save the figure
            vocab_size: Vocabulary size (if None, inferred from data)
            
        Returns:
            Matplotlib figure
        """
        # Get the number of time steps
        if not particles or not particles[0].token_probs_history:
            return plt.figure()
            
        time_steps = len(particles[0].token_probs_history)
        
        # Determine vocabulary size
        if vocab_size is None:
            # Find all unique tokens across all particles and time steps
            all_tokens = set()
            for particle in particles:
                for step_probs in particle.token_probs_history:
                    all_tokens.update(step_probs.keys())
            vocab_size = max(all_tokens) + 1 if all_tokens else 32000  # Default vocab size
        
        # Create probability matrix (tokens x time)
        prob_matrix = np.zeros((vocab_size, time_steps))
        
        # Aggregate probabilities across particles
        for particle in particles:
            for t, step_probs in enumerate(particle.token_probs_history):
                for token_id, prob in step_probs.items():
                    if token_id < vocab_size:
                        prob_matrix[token_id, t] += prob
        
        # Normalize by number of particles
        prob_matrix /= len(particles)
        
        # Create figure with appropriate aspect ratio
        fig_height = min(20, max(8, vocab_size / 1000))  # Scale height with vocab size
        fig, ax = plt.subplots(figsize=(self.figsize[0], fig_height))
        
        # Plot heatmap (only show non-zero probabilities)
        # Use log scale for better visibility
        log_probs = np.log10(prob_matrix + 1e-10)  # Add small constant to avoid log(0)
        
        # Only show rows (tokens) that have any non-zero probability
        active_tokens = np.any(prob_matrix > 1e-6, axis=1)
        active_indices = np.where(active_tokens)[0]
        
        if len(active_indices) > 0:
            # Create compressed view showing only active tokens
            compressed_probs = log_probs[active_indices, :]
            
            im = ax.imshow(
                compressed_probs,
                aspect='auto',
                cmap='hot',
                interpolation='nearest',
                vmin=-6,  # 10^-6 probability
                vmax=0    # 10^0 = 1.0 probability
            )
            
            # Set labels
            ax.set_xlabel("Time Step", fontsize=12)
            ax.set_ylabel(f"Tokens (showing {len(active_indices)} of {vocab_size})", fontsize=12)
            
            # Set x-ticks
            ax.set_xticks(range(0, time_steps, max(1, time_steps // 10)))
            
            # Don't show y-ticks (too many)
            ax.set_yticks([])
        else:
            # Fallback if no active tokens
            ax.text(0.5, 0.5, "No token probabilities found", 
                   transform=ax.transAxes, ha='center', va='center')
        
        ax.set_title(
            f"Token Probability Distribution: \"{prompt}\"",
            fontsize=14,
            fontweight='bold'
        )
        
        # Add colorbar
        if len(active_indices) > 0:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Log10(Probability)", fontsize=10)
            
            # Set colorbar ticks
            cbar.set_ticks([-6, -4, -2, 0])
            cbar.set_ticklabels(['10⁻⁶', '10⁻⁴', '10⁻²', '1'])
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.save_dpi, bbox_inches='tight')

        return fig

    def visualize_bumplot(
        self,
        particles: List[Particle],
        output_path: Optional[str] = None,
        max_vocab_display: int = 100,
        color_by: str = 'transition_prob',
        show_tokens: bool = True,
        curve_force: float = 0.5,
        figsize: Optional[Tuple[int, int]] = None,
        **kwargs
    ) -> plt.Figure:
        """
        Create bump plot visualization of particle token trajectories.

        Args:
            particles: List of particles from filter
            output_path: Path to save figure
            max_vocab_display: Maximum number of unique tokens to display
            color_by: Method for coloring curves ('transition_prob', 'particle_id', 'entropy')
            show_tokens: Whether to show token text labels
            curve_force: Intensity of curve bending (0-1)
            figsize: Figure size override
            **kwargs: Additional args passed to bumplot

        Returns:
            Matplotlib figure with bump plot
        """
        if not particles:
            logger.warning("No particles provided for bumplot visualization")
            return plt.figure()

        # Prepare data for bumplot
        df, metadata = self._prepare_bumplot_data(particles, max_vocab_display)

        if df.empty:
            logger.warning("No data to visualize in bumplot")
            return plt.figure()

        # Set up figure
        if figsize is None:
            figsize = self.figsize
        fig, ax = plt.subplots(figsize=figsize)

        # Get color mapping based on selected method
        colors = self._get_bumplot_colors(particles, metadata, color_by)

        # Get particle column names (excluding 'timestep')
        particle_columns = [col for col in df.columns if col.startswith('particle_')]

        # Create bumplot
        try:
            bumplot(
                x='timestep',
                y_columns=particle_columns,
                data=df,
                curve_force=curve_force,
                colors=colors,
                ax=ax,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error creating bumplot: {e}")
            return fig

        # Customize plot
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Token Rank', fontsize=12)
        ax.set_title('Particle Token Trajectories (Bump Plot)', fontsize=14, fontweight='bold')

        # Add token labels if requested
        if show_tokens and 'token_to_text' in metadata:
            self._add_token_labels(ax, metadata['token_to_text'], metadata['token_ranks'])

        # Add colorbar if using probability coloring
        if color_by == 'transition_prob' and 'colormap' in metadata:
            self._add_probability_colorbar(fig, metadata['colormap'])

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=self.save_dpi, bbox_inches='tight')

        return fig

    def _prepare_bumplot_data(
        self,
        particles: List[Particle],
        max_vocab_display: int
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Transform particle data into bumplot-compatible format.

        Returns:
            - DataFrame with columns: ['timestep', 'particle_0', 'particle_1', ...]
            - Metadata dict with probability and token info
        """
        if not particles:
            return pd.DataFrame(), {}

        # Find max sequence length
        max_length = max(len(p.tokens) for p in particles)

        # Get token ranks
        token_ranks = self._rank_tokens_by_frequency(particles, max_vocab_display)

        # Create DataFrame structure
        data = {'timestep': list(range(max_length))}

        # Track metadata
        metadata = {
            'token_ranks': token_ranks,
            'transition_probs': {},
            'token_to_text': {}
        }

        # Process each particle
        for i, particle in enumerate(particles):
            col_name = f'particle_{i}'

            # Convert tokens to ranks, padding with NaN for shorter sequences
            ranked_tokens = []
            for t, token in enumerate(particle.tokens):
                if token in token_ranks:
                    ranked_tokens.append(token_ranks[token])

                    # Store transition probability if available
                    if t > 0 and t-1 < len(particle.token_probs_history):
                        if token in particle.token_probs_history[t-1]:
                            prob = particle.token_probs_history[t-1][token]
                            metadata['transition_probs'][(i, t)] = prob
                else:
                    # Token not in top-k, assign worst rank
                    ranked_tokens.append(max_vocab_display + 1)

            # Pad with NaN for shorter sequences
            ranked_tokens += [np.nan] * (max_length - len(ranked_tokens))
            data[col_name] = ranked_tokens

        # Store token text mappings
        if hasattr(self, 'tokenizer'):
            for token_id, rank in token_ranks.items():
                try:
                    token_text = self.tokenizer.decode([token_id])
                    metadata['token_to_text'][rank] = token_text[:20]  # Truncate long tokens
                except:
                    metadata['token_to_text'][rank] = f"<token_{token_id}>"

        return pd.DataFrame(data), metadata

    def _rank_tokens_by_frequency(
        self,
        particles: List[Particle],
        max_vocab_display: int
    ) -> Dict[int, int]:
        """
        Rank tokens by their overall frequency across all particles.
        Most frequent = rank 1.

        Returns:
            Dict mapping token_id to rank
        """
        token_counts = {}

        # Count token occurrences
        for particle in particles:
            for token in particle.tokens:
                token_counts[token] = token_counts.get(token, 0) + 1

        # Sort by count and assign ranks
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)

        # Limit to max_vocab_display
        sorted_tokens = sorted_tokens[:max_vocab_display]

        return {token: rank+1 for rank, (token, _) in enumerate(sorted_tokens)}

    def _calculate_path_probabilities(
        self,
        particles: List[Particle]
    ) -> Dict[Tuple[int, int], float]:
        """
        Calculate path probabilities as fraction of particles.

        Returns:
            Dict mapping (timestep, token_id) to probability
        """
        path_counts = {}

        for t in range(max(len(p.tokens) for p in particles)):
            tokens_at_t = {}

            for particle in particles:
                if t < len(particle.tokens):
                    token = particle.tokens[t]
                    tokens_at_t[token] = tokens_at_t.get(token, 0) + 1

            # Normalize to get probabilities
            total = len([p for p in particles if t < len(p.tokens)])
            if total > 0:
                for token, count in tokens_at_t.items():
                    path_counts[(t, token)] = count / total

        return path_counts

    def _get_bumplot_colors(
        self,
        particles: List[Particle],
        metadata: Dict,
        color_by: str
    ) -> List[str]:
        """
        Generate colors for bumplot based on selected method.

        Returns:
            List of hex color strings
        """
        n_particles = len(particles)

        if color_by == 'transition_prob':
            # Color based on average transition probability
            colors = []
            cmap = plt.cm.plasma

            for i in range(n_particles):
                # Get all transition probs for this particle
                probs = [metadata['transition_probs'].get((i, t), 0.5)
                        for t in range(len(particles[i].tokens))]

                # Use average probability for color
                avg_prob = np.mean(probs) if probs else 0.5
                color = cmap(avg_prob)
                colors.append(plt.matplotlib.colors.to_hex(color))

            metadata['colormap'] = cmap

        elif color_by == 'entropy':
            # Color based on average entropy
            colors = []
            cmap = plt.cm.viridis

            for particle in particles:
                entropy_scores = particle.entropy_scores
                avg_entropy = np.mean(entropy_scores) if entropy_scores else 0.5
                # Normalize entropy (assuming max ~3.5 for typical vocab)
                norm_entropy = min(avg_entropy / 3.5, 1.0)
                color = cmap(norm_entropy)
                colors.append(plt.matplotlib.colors.to_hex(color))

        elif color_by == 'particle_id':
            # Use distinct colors for each particle
            cmap = plt.cm.tab20 if n_particles <= 20 else plt.cm.hsv
            colors = [plt.matplotlib.colors.to_hex(cmap(i / n_particles))
                     for i in range(n_particles)]
        else:
            # Default: use a gradient
            colors = ['#' + ''.join([f'{int(255*(1-i/n_particles)):02x}' if j==0
                                    else f'{int(100+155*i/n_particles):02x}'
                                    for j in range(3)])
                     for i in range(n_particles)]

        return colors

    def _add_token_labels(
        self,
        ax: plt.Axes,
        token_to_text: Dict[int, str],
        token_ranks: Dict[int, int]
    ):
        """
        Add token text labels to y-axis.
        """
        if not token_to_text:
            return

        # Get current y-ticks
        y_ticks = ax.get_yticks()

        # Map ticks to token text
        y_labels = []
        for tick in y_ticks:
            if int(tick) in token_to_text:
                y_labels.append(token_to_text[int(tick)])
            else:
                y_labels.append('')

        ax.set_yticklabels(y_labels, fontsize=8)

    def _add_probability_colorbar(
        self,
        fig: plt.Figure,
        cmap
    ):
        """
        Add a colorbar showing probability scale.
        """
        # Create a ScalarMappable for the colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])

        # Add colorbar
        cbar = fig.colorbar(sm, ax=fig.axes[0], pad=0.02)
        cbar.set_label('Transition Probability', fontsize=10)
        cbar.ax.tick_params(labelsize=8)
