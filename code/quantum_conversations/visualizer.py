"""
Visualization tools for token sequences and particle paths.

Provides multiple visualization types:
- Bumplot diagrams for token generation paths
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
        **kwargs
    ) -> plt.Figure:
        """
        Deprecated: Use visualize_bumplot instead.
        This method now redirects to visualize_bumplot for backwards compatibility.
        """
        return self.visualize_bumplot(
            particles=particles,
            prompt=prompt,
            output_path=output_path,
            **kwargs
        )

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
        max_vocab_display: int = 15,
        color_by: str = 'transition_prob',
        show_tokens: bool = True,
        curve_force: float = 0.5,
        figsize: Optional[Tuple[int, int]] = None,
        colormap: str = 'RdYlGn',
        prompt: Optional[str] = None,
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
            colormap: Colormap for probability display
            prompt: Original prompt for title
            **kwargs: Additional args

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

        # Set up figure with proper layout
        if figsize is None:
            figsize = (18, 10)  # Default size for good visibility

        # Create figure with gridspec for better layout control
        fig = plt.figure(figsize=figsize)
        from matplotlib.gridspec import GridSpec

        # Create grid: main plot + narrow colorbar space
        gs = GridSpec(1, 2, width_ratios=[30, 1], figure=fig, wspace=0.02)
        ax = fig.add_subplot(gs[0, 0])

        # Use custom bumplot with smooth curves
        try:
            from .custom_bumplot import create_custom_bumplot, add_token_labels

            # Create custom bumplot with smooth splines
            create_custom_bumplot(
                df,
                metadata,
                ax,
                colormap=colormap,
                alpha=0.6,
                linewidth=1.0,
                curve_force=curve_force
            )

            metadata['colormap_name'] = colormap

        except Exception as e:
            logger.warning(f"Using fallback bumplot: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to simple plotting
            particle_columns = [col for col in df.columns if col.startswith('particle_')]
            for col in particle_columns:
                trajectory = df[['timestep', col]].dropna()
                if len(trajectory) > 1:
                    ax.plot(trajectory['timestep'], trajectory[col],
                           alpha=0.3, linewidth=0.5, color='steelblue')

        # Set proper axis limits based on actual data
        max_timestep = metadata.get('max_length', len(df)) - 1
        ax.set_xlim(-0.5, max_timestep + 0.5)

        # Calculate actual ranks used
        ranks_used = set()
        for col in df.columns:
            if col.startswith('particle_'):
                ranks_used.update(df[col].dropna().unique())

        if ranks_used:
            # Filter out any out-of-range ranks
            valid_ranks = [r for r in ranks_used if 1 <= r <= max_vocab_display]
            if valid_ranks:
                max_rank_used = int(max(valid_ranks))
                # Set y limits with rank 1 at top, more padding to prevent cutoff
                ax.set_ylim(max_rank_used + 0.8, 0.3)  # More padding for tokens

                # Set reasonable number of y-ticks
                n_ticks = min(max_rank_used, max_vocab_display)
                ax.set_yticks(range(1, n_ticks + 1))
                ax.set_yticklabels(range(1, n_ticks + 1))
            else:
                ax.set_ylim(max_vocab_display + 0.8, 0.3)
                ax.set_yticks(range(1, min(max_vocab_display + 1, 16)))
        else:
            ax.set_ylim(max_vocab_display + 0.8, 0.3)
            ax.set_yticks(range(1, min(max_vocab_display + 1, 16)))

        # Customize plot appearance with much larger fonts
        ax.set_xlabel('Output position', fontsize=18, fontweight='normal')
        ax.set_ylabel('Token rank (by frequency)', fontsize=18, fontweight='normal')
        ax.tick_params(axis='both', labelsize=14)  # Larger tick labels

        # No title per requirements

        # Add grid for better readability
        ax.grid(True, axis='x', alpha=0.3, linestyle=':')
        ax.grid(True, axis='y', alpha=0.2, linestyle=':')

        # Add token labels if requested
        if show_tokens:
            try:
                from .custom_bumplot import add_token_labels
                add_token_labels(
                    ax,
                    metadata,
                    self.tokenizer,
                    colormap=colormap,
                    show_freq=False,  # No frequency labels
                    min_freq_threshold=0.0,  # Show all tokens that have ranks
                    max_labels_per_timestep=max_vocab_display  # Show all ranked tokens
                )
            except Exception as e:
                logger.warning(f"Token label error: {e}")
                # Use simplified fallback
                self._overlay_token_labels(ax, df, metadata, max_length=max_timestep + 1)

        # Add proper dual probability legend
        if color_by == 'transition_prob' or show_tokens:
            cbar_ax = fig.add_subplot(gs[0, 1])
            self._add_dual_probability_legend_improved(fig, cbar_ax, colormap)

        # Adjust layout to prevent overlap
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
        Ranks tokens at each position by cross-particle frequency.
        Ensures particles with identical tokens converge to same rank.

        Returns:
            - DataFrame with columns: ['timestep', 'particle_0', 'particle_1', ...]
            - Metadata dict with probability and token info
        """
        if not particles:
            return pd.DataFrame(), {}

        from collections import Counter, defaultdict

        # Find max sequence length
        max_length = max(len(p.tokens) for p in particles)

        # Collect token data at each position
        position_data = defaultdict(lambda: defaultdict(list))

        for particle_idx, particle in enumerate(particles):
            for t in range(min(len(particle.tokens), max_length)):
                token_id = particle.tokens[t]

                # Get within-particle probability if available
                # For prompt tokens (first few positions), probability should be 1.0
                prompt_length = len(particle.prompt_tokens) if hasattr(particle, 'prompt_tokens') else 0

                if t < prompt_length:
                    # Prompt tokens have 100% probability (they are observed)
                    prob = 1.0
                elif t > 0 and t-1 < len(particle.token_probs_history):
                    if token_id in particle.token_probs_history[t-1]:
                        prob = particle.token_probs_history[t-1][token_id]
                    else:
                        prob = 0.0
                else:
                    prob = 1.0  # First generated token

                position_data[t][token_id].append({
                    'particle_idx': particle_idx,
                    'within_particle_prob': prob
                })

        # Compute cross-particle ranks at each position
        position_ranks = {}
        token_metadata = {}
        token_to_text = {}  # Cache for decoded tokens
        transition_probs = {}  # Store transition probabilities

        for t in range(max_length):
            # Count frequency of each token at this position
            token_counts = Counter()
            for token_id, particle_list in position_data[t].items():
                token_counts[token_id] = len(particle_list)

            # Rank tokens by frequency (most common = rank 1)
            # Use stable sort to ensure consistent ranking
            ranked_tokens = sorted(token_counts.items(), key=lambda x: (-x[1], x[0]))

            # Only keep top max_vocab_display tokens
            ranked_tokens = ranked_tokens[:max_vocab_display]

            # Assign ranks - particles with same token MUST get same rank
            rank_map = {}
            for rank, (token_id, count) in enumerate(ranked_tokens, 1):
                rank_map[token_id] = rank

                # Decode token text (cache for efficiency)
                if token_id not in token_to_text:
                    try:
                        token_text = self.tokenizer.decode([token_id])
                        # Escape special characters that might be interpreted as LaTeX
                        token_text = token_text.replace('$', r'\$').replace('\\', r'\\\\')
                        # Clean up token text
                        token_text = token_text.strip().replace('\n', '[nl]').replace('\t', '[tab]')
                        # Better truncation that preserves meaning
                        if len(token_text) > 10:
                            token_text = token_text[:8] + '..'
                    except Exception as e:
                        token_text = f"<{token_id}>"
                    token_to_text[token_id] = token_text
                else:
                    token_text = token_to_text[token_id]

                # Calculate average within-particle probability
                avg_prob = np.mean([
                    p['within_particle_prob']
                    for p in position_data[t][token_id]
                ])

                # Store metadata for this token at this position
                token_metadata[(t, rank)] = {
                    'token_id': token_id,
                    'token_text': token_text,
                    'frequency': count,
                    'frequency_pct': count / len(particles),
                    'avg_within_prob': avg_prob,
                    'particle_indices': [p['particle_idx'] for p in position_data[t][token_id]]
                }

            position_ranks[t] = rank_map

        # Calculate transition probabilities between timesteps
        for t in range(max_length - 1):
            transition_counts = Counter()
            for particle_idx, particle in enumerate(particles):
                if t < len(particle.tokens) - 1:
                    from_token = particle.tokens[t]
                    to_token = particle.tokens[t + 1]

                    from_rank = position_ranks[t].get(from_token, max_vocab_display + 1)
                    to_rank = position_ranks[t + 1].get(to_token, max_vocab_display + 1)

                    if from_rank <= max_vocab_display and to_rank <= max_vocab_display:
                        transition_counts[(t, from_rank, to_rank)] += 1

            # Normalize to get probabilities
            for key, count in transition_counts.items():
                transition_probs[key] = count / len(particles)

        # Create DataFrame with particle trajectories
        data = {'timestep': list(range(max_length))}

        # Track each particle's path through the ranks
        # IMPORTANT: Particles with identical tokens must have identical trajectories
        for particle_idx, particle in enumerate(particles):
            trajectory = []
            for t in range(max_length):
                if t < len(particle.tokens):
                    token_id = particle.tokens[t]
                    # Use the rank map to ensure consistency
                    rank = position_ranks[t].get(token_id, max_vocab_display + 1)
                    # Clamp rank to max_vocab_display to prevent out-of-range curves
                    if rank > max_vocab_display:
                        rank = max_vocab_display  # Put overflow at the max visible rank
                    trajectory.append(float(rank))  # Use float for smoother curves
                else:
                    trajectory.append(np.nan)

            data[f'particle_{particle_idx}'] = trajectory

        df = pd.DataFrame(data)

        metadata = {
            'token_metadata': token_metadata,
            'position_ranks': position_ranks,
            'n_particles': len(particles),
            'max_length': max_length,
            'max_vocab_display': max_vocab_display,
            'token_to_text': token_to_text,
            'transition_probs': transition_probs,
            'token_ranks': {token: rank for ranks in position_ranks.values()
                          for token, rank in ranks.items()}  # Flattened rank map
        }

        return df, metadata

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
            # Use a uniform color since probabilities are now in token backgrounds
            colors = ['#4682B4'] * n_particles  # Steel blue

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

    def _overlay_token_labels(
        self,
        ax: plt.Axes,
        df: pd.DataFrame,
        metadata: Dict,
        max_length: int
    ):
        """
        Overlay token text at each (timestep, rank) position on the plot.
        Background color indicates within-particle probability.
        """
        token_metadata = metadata.get('token_metadata', {})
        if not token_metadata:
            return

        import matplotlib.cm as cm
        from matplotlib.colors import to_hex

        # Track positions to avoid overlap
        used_positions = set()

        for (timestep, rank), info in token_metadata.items():
            if timestep < max_length:
                # Skip if frequency too low (less than 5% of particles)
                if info['frequency_pct'] < 0.05 and rank > 10:
                    continue

                # Determine background color based on within-particle probability
                avg_prob = info['avg_within_prob']

                # Use color gradient: high prob = green, low prob = red
                if avg_prob > 0.7:
                    color = cm.YlGn(0.7 + avg_prob * 0.3)
                elif avg_prob > 0.3:
                    color = cm.YlOrRd(1.0 - avg_prob)
                else:
                    color = cm.Reds(0.3 + avg_prob * 0.7)

                # Adjust alpha based on frequency
                alpha = min(0.5 + info['frequency_pct'] * 0.5, 0.95)

                # Check for text overlap
                y_offset = 0
                key = (timestep, rank)
                nearby_keys = [(timestep, rank + i) for i in range(-1, 2) if i != 0]
                for nkey in nearby_keys:
                    if nkey in used_positions:
                        y_offset = 0.2  # Slight offset if nearby position is used
                        break

                # Add text with colored background
                text = ax.text(
                    timestep,
                    rank + y_offset,
                    info['token_text'],
                    fontsize=7 if info['frequency_pct'] > 0.1 else 6,
                    ha='center',
                    va='center',
                    weight='bold' if info['frequency_pct'] > 0.5 else 'normal',
                    bbox=dict(
                        boxstyle='round,pad=0.25',
                        facecolor=to_hex(color),
                        edgecolor='darkgray',
                        alpha=alpha,
                        linewidth=0.5
                    ),
                    zorder=1000 + rank  # Layer by rank
                )

                # Add frequency annotation for significant tokens
                if info['frequency_pct'] > 0.2:
                    ax.text(
                        timestep,
                        rank + y_offset - 0.3,
                        f"{info['frequency_pct']:.0%}",
                        fontsize=5,
                        ha='center',
                        va='top',
                        color='gray',
                        alpha=0.7,
                        zorder=999
                    )

                used_positions.add(key)

    def _add_dual_probability_legend_improved(
        self,
        fig: plt.Figure,
        cbar_ax: plt.Axes,
        colormap: str
    ):
        """
        Add simplified colorbar - narrow, short, minimal labels.
        """
        from matplotlib.colors import Normalize
        from matplotlib.colorbar import ColorbarBase
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        cmap = plt.colormaps[colormap]
        norm = Normalize(vmin=0, vmax=1)

        # Create a smaller colorbar positioned at bottom 1/3 of the figure
        # Use the original cbar_ax parent to get the figure
        parent_ax = cbar_ax

        # Clear the full-height axis
        cbar_ax.set_visible(False)

        # Get main axis position for alignment
        main_axes = [ax for ax in fig.get_axes() if ax != cbar_ax]
        if main_axes:
            main_ax = main_axes[0]
            ax_pos = main_ax.get_position()
        else:
            ax_pos = None

        # Create new smaller axis for colorbar with small gap from main plot
        # Position at bottom 1/3, narrow, with appropriate spacing
        small_cbar_ax = fig.add_axes([0.88, 0.15, 0.008, 0.25])  # [left, bottom, width, height]

        # Create colorbar in the smaller axis
        cbar = ColorbarBase(
            small_cbar_ax,
            cmap=cmap,
            norm=norm,
            orientation='vertical'
        )

        # Remove inline tick labels
        cbar.set_ticks([0, 1])
        cbar.ax.set_yticklabels([])  # No inline labels
        cbar.ax.tick_params(size=0)  # Hide tick marks

        # Add labels above and below colorbar
        cbar_pos = small_cbar_ax.get_position()
        label_x = cbar_pos.x0 + cbar_pos.width / 2

        # Position 0% label aligned with x-axis if possible
        if ax_pos:
            label_y_bottom = ax_pos.y0
        else:
            label_y_bottom = cbar_pos.y0 - 0.02

        fig.text(label_x, label_y_bottom, '0%', ha='center', va='bottom', fontsize=10)

        # Position 100% label above colorbar
        fig.text(label_x, cbar_pos.y1 + 0.01, '100%', ha='center', va='bottom', fontsize=10)

    def _add_dual_probability_legend(
        self,
        fig: plt.Figure,
        colormap: str
    ):
        """
        Legacy method for backwards compatibility.
        """
        from matplotlib.colors import Normalize
        import matplotlib.pyplot as plt

        cmap = plt.colormaps[colormap]
        norm = Normalize(vmin=0, vmax=1)

        # Create colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # Try to find available space for colorbar
        try:
            # Add colorbar to the right of the plot
            cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])
            cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
            cbar.set_label('Probability', fontsize=10)

            # Add explanatory text
            fig.text(0.91, 0.75, 'Colors:', fontsize=9, ha='right', weight='bold')
            fig.text(0.91, 0.72, 'Curves: transition freq', fontsize=8, ha='right')
            fig.text(0.91, 0.70, 'Tokens: confidence', fontsize=8, ha='right')
        except:
            # Fallback if figure layout doesn't support it
            pass

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
