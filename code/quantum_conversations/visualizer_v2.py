"""
Enhanced bumplot visualization for Quantum Conversations.
This version properly handles token convergence and dual ranking systems.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)

class EnhancedBumplotVisualizer:
    """Enhanced bumplot with proper token convergence and dual ranking."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def prepare_bumplot_data(
        self,
        particles: List,
        max_timesteps: Optional[int] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Prepare data where particles with same tokens converge to same y-position.

        Returns:
            - DataFrame with particle trajectories
            - Metadata including token info and probabilities
        """
        if not particles:
            return pd.DataFrame(), {}

        # Determine sequence length
        if max_timesteps is None:
            max_timesteps = max(len(p.tokens) for p in particles)
        else:
            max_timesteps = min(max_timesteps, max(len(p.tokens) for p in particles))

        # Collect tokens at each position and their within-particle probabilities
        position_data = defaultdict(lambda: defaultdict(list))

        for particle_idx, particle in enumerate(particles):
            for t in range(min(len(particle.tokens), max_timesteps)):
                token_id = particle.tokens[t]

                # Get within-particle probability if available
                if t > 0 and t-1 < len(particle.token_probs_history):
                    if token_id in particle.token_probs_history[t-1]:
                        prob = particle.token_probs_history[t-1][token_id]
                    else:
                        prob = 0.0  # Token wasn't in the distribution (shouldn't happen)
                else:
                    prob = 1.0  # First token or no history

                position_data[t][token_id].append({
                    'particle_idx': particle_idx,
                    'within_particle_prob': prob
                })

        # Compute cross-particle ranks at each position
        position_ranks = {}
        position_tokens = {}
        token_metadata = defaultdict(dict)

        for t in range(max_timesteps):
            # Count frequency of each token at this position
            token_counts = Counter()
            for token_id, particle_list in position_data[t].items():
                token_counts[token_id] = len(particle_list)

            # Rank tokens by frequency (most common = rank 1)
            ranked_tokens = sorted(token_counts.items(), key=lambda x: (-x[1], x[0]))

            # Assign ranks
            rank_map = {}
            for rank, (token_id, count) in enumerate(ranked_tokens, 1):
                rank_map[token_id] = rank

                # Decode token text
                try:
                    token_text = self.tokenizer.decode([token_id])
                    # Clean up token text
                    token_text = token_text.strip().replace('\n', '↵').replace('\t', '→')
                    if len(token_text) > 12:
                        token_text = token_text[:10] + '..'
                except:
                    token_text = f"<{token_id}>"

                # Store metadata for this token at this position
                token_metadata[(t, rank)] = {
                    'token_id': token_id,
                    'token_text': token_text,
                    'frequency': count,
                    'frequency_pct': count / len(particles),
                    'avg_within_prob': np.mean([
                        p['within_particle_prob']
                        for p in position_data[t][token_id]
                    ])
                }

            position_ranks[t] = rank_map
            position_tokens[t] = ranked_tokens

        # Create DataFrame with particle trajectories
        data = {'timestep': list(range(max_timesteps))}

        # Track each particle's path through the ranks
        for particle_idx, particle in enumerate(particles):
            trajectory = []
            for t in range(max_timesteps):
                if t < len(particle.tokens):
                    token_id = particle.tokens[t]
                    rank = position_ranks[t].get(token_id, len(position_ranks[t]) + 1)
                    trajectory.append(rank)
                else:
                    trajectory.append(np.nan)

            data[f'particle_{particle_idx}'] = trajectory

        df = pd.DataFrame(data)

        metadata = {
            'token_metadata': token_metadata,
            'position_ranks': position_ranks,
            'position_tokens': position_tokens,
            'n_particles': len(particles),
            'max_timesteps': max_timesteps
        }

        return df, metadata

    def create_enhanced_bumplot(
        self,
        particles: List,
        max_timesteps: Optional[int] = 20,
        figsize: Tuple[int, int] = (18, 12),
        show_tokens: bool = True,
        color_by_prob: bool = True,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Create enhanced bumplot with proper token convergence.

        Args:
            particles: List of particles from filter
            max_timesteps: Maximum timesteps to show
            figsize: Figure size
            show_tokens: Whether to overlay token labels
            color_by_prob: Color token backgrounds by within-particle probability
            title: Plot title

        Returns:
            Matplotlib figure
        """
        # Prepare data
        df, metadata = self.prepare_bumplot_data(particles, max_timesteps)

        if df.empty:
            return plt.figure()

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Get particle columns
        particle_cols = [c for c in df.columns if c.startswith('particle_')]
        n_particles = len(particle_cols)

        # Plot each particle's trajectory
        for col in particle_cols:
            trajectory = df[['timestep', col]].dropna()
            if len(trajectory) > 1:
                # Create smooth curve through points
                x = trajectory['timestep'].values
                y = trajectory[col].values

                # Add some curve smoothing
                from scipy.interpolate import make_interp_spline
                try:
                    # Try to create smooth spline
                    if len(x) >= 4:
                        k = min(3, len(x) - 1)
                        spl = make_interp_spline(x, y, k=k)
                        x_smooth = np.linspace(x.min(), x.max(), 100)
                        y_smooth = spl(x_smooth)
                    else:
                        x_smooth, y_smooth = x, y

                    # Plot with low opacity for individual paths
                    ax.plot(x_smooth, y_smooth, alpha=0.15, linewidth=0.5, color='steelblue')
                except:
                    # Fallback to simple line
                    ax.plot(x, y, alpha=0.15, linewidth=0.5, color='steelblue')

        # Add token labels if requested
        if show_tokens:
            self._add_token_labels(ax, metadata, color_by_prob)

        # Customize axes
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Token Rank (by frequency)', fontsize=12)

        # Set y-axis (rank 1 at top)
        max_rank = df[particle_cols].max().max()
        if not np.isnan(max_rank):
            ax.set_ylim(max_rank + 0.5, 0.5)
            ax.set_yticks(range(1, int(max_rank) + 1))

        # Set x-axis
        ax.set_xlim(-0.5, max_timesteps - 0.5)
        ax.set_xticks(range(max_timesteps))

        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')

        # Title
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        else:
            ax.set_title(f'Token Trajectories ({n_particles} particles)', fontsize=14, fontweight='bold', pad=20)

        # Add legend for probability coloring
        if show_tokens and color_by_prob:
            self._add_probability_legend(fig)

        plt.tight_layout()
        return fig

    def _add_token_labels(self, ax, metadata, color_by_prob=True):
        """Add token labels with proper spacing and coloring."""
        token_metadata = metadata['token_metadata']

        # Track positions to avoid overlap
        used_positions = defaultdict(list)

        for (timestep, rank), info in token_metadata.items():
            # Check for overlaps
            y_offset = 0
            for used_y in used_positions[timestep]:
                if abs(rank - used_y) < 0.3:  # Too close
                    y_offset = 0.15 if rank > used_y else -0.15

            # Determine background color based on within-particle probability
            if color_by_prob:
                prob = info['avg_within_prob']
                # Color scale: high prob = green, low prob = red
                if prob > 0.5:
                    face_color = plt.cm.RdYlGn(prob)
                elif prob > 0.1:
                    face_color = plt.cm.RdYlGn(prob)
                else:
                    face_color = plt.cm.RdYlGn(0.1)

                # Adjust alpha based on frequency
                alpha = min(0.3 + info['frequency_pct'] * 0.7, 1.0)
            else:
                face_color = 'white'
                alpha = 0.8

            # Add text with background
            text = ax.text(
                timestep,
                rank + y_offset,
                info['token_text'],
                fontsize=8,
                ha='center',
                va='center',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor=face_color,
                    edgecolor='gray',
                    alpha=alpha,
                    linewidth=0.5
                ),
                zorder=1000
            )

            # Add frequency annotation if significant
            if info['frequency_pct'] > 0.1:  # More than 10% of particles
                ax.text(
                    timestep,
                    rank + y_offset - 0.25,
                    f"{info['frequency_pct']:.0%}",
                    fontsize=6,
                    ha='center',
                    va='top',
                    color='gray',
                    zorder=999
                )

            used_positions[timestep].append(rank + y_offset)

    def _add_probability_legend(self, fig):
        """Add legend explaining the probability coloring."""
        # Create color patches for legend
        high_prob = mpatches.Patch(color=plt.cm.RdYlGn(0.9), label='High within-particle prob')
        med_prob = mpatches.Patch(color=plt.cm.RdYlGn(0.5), label='Medium within-particle prob')
        low_prob = mpatches.Patch(color=plt.cm.RdYlGn(0.1), label='Low within-particle prob')

        # Add legend
        legend = fig.legend(
            handles=[high_prob, med_prob, low_prob],
            loc='upper right',
            bbox_to_anchor=(0.98, 0.98),
            fontsize=10,
            title='Token Background Color',
            title_fontsize=10
        )
        legend.get_frame().set_alpha(0.8)