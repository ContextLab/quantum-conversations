"""
Custom bumplot implementation with per-segment coloring for Quantum Conversations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict, Counter
from scipy.interpolate import make_interp_spline
import logging

logger = logging.getLogger(__name__)


def create_custom_bumplot(
    df: pd.DataFrame,
    metadata: Dict,
    ax: plt.Axes,
    colormap: str = 'RdYlGn',
    alpha: float = 0.7,
    linewidth: float = 1.5
) -> None:
    """
    Create custom bumplot with per-segment coloring.

    Args:
        df: DataFrame with particle trajectories
        metadata: Metadata including transition info
        ax: Matplotlib axes
        colormap: Colormap to use for probabilities
        alpha: Line transparency
        linewidth: Line width
    """
    from matplotlib.cm import get_cmap
    from matplotlib.colors import to_hex

    cmap = get_cmap(colormap)
    norm = Normalize(vmin=0, vmax=1)

    # Get transition frequencies from metadata
    transition_freqs = metadata.get('transition_frequencies', {})

    # Plot each particle trajectory
    particle_cols = [c for c in df.columns if c.startswith('particle_')]

    for col in particle_cols:
        trajectory = df[['timestep', col]].dropna()

        if len(trajectory) < 2:
            continue

        x = trajectory['timestep'].values
        y = trajectory[col].values

        # Create segments with colors based on transition frequency
        for i in range(len(x) - 1):
            x_seg = [x[i], x[i+1]]
            y_seg = [y[i], y[i+1]]

            # Get transition frequency for this segment
            from_rank = int(y[i])
            to_rank = int(y[i+1])
            timestep = int(x[i])

            # Look up transition frequency
            transition_key = (timestep, from_rank, to_rank)
            freq = transition_freqs.get(transition_key, 0.0)

            # Determine color based on frequency
            color = cmap(norm(freq))

            # Create smooth curve between points
            if abs(y_seg[1] - y_seg[0]) > 0.01:  # Only curve if there's movement
                try:
                    # Create smooth interpolation
                    x_smooth = np.linspace(x_seg[0], x_seg[1], 20)
                    # Add curvature
                    t = np.linspace(0, 1, 20)
                    curve_factor = 0.5  # How much to curve
                    x_curved = x_seg[0] + t * (x_seg[1] - x_seg[0])
                    y_curved = y_seg[0] + t * (y_seg[1] - y_seg[0])
                    # Add sinusoidal curve
                    offset = curve_factor * np.sin(t * np.pi) * (x_seg[1] - x_seg[0]) * 0.3
                    x_curved = x_curved + offset * np.sign(y_seg[1] - y_seg[0]) * 0.2

                    ax.plot(x_curved, y_curved, color=color, alpha=alpha,
                           linewidth=linewidth * (0.5 + freq * 0.5))
                except:
                    # Fallback to straight line
                    ax.plot(x_seg, y_seg, color=color, alpha=alpha, linewidth=linewidth)
            else:
                # Straight line for no rank change
                ax.plot(x_seg, y_seg, color=color, alpha=alpha, linewidth=linewidth)


def prepare_transition_data(
    particles: List,
    max_timesteps: Optional[int] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Prepare data with transition frequency information.

    Returns:
        - DataFrame with particle trajectories
        - Metadata including transition frequencies
    """
    from collections import Counter, defaultdict

    if not particles:
        return pd.DataFrame(), {}

    # Determine sequence length
    if max_timesteps is None:
        max_timesteps = max(len(p.tokens) for p in particles)
    else:
        max_timesteps = min(max_timesteps, max(len(p.tokens) for p in particles))

    # First pass: collect token frequencies at each position
    position_data = defaultdict(lambda: defaultdict(list))

    for particle_idx, particle in enumerate(particles):
        for t in range(min(len(particle.tokens), max_timesteps)):
            token_id = particle.tokens[t]

            # Get within-particle probability
            if t > 0 and t-1 < len(particle.token_probs_history):
                if token_id in particle.token_probs_history[t-1]:
                    prob = particle.token_probs_history[t-1][token_id]
                else:
                    prob = 0.0
            else:
                prob = 1.0

            position_data[t][token_id].append({
                'particle_idx': particle_idx,
                'within_particle_prob': prob
            })

    # Compute ranks at each position
    position_ranks = {}
    token_metadata = {}

    for t in range(max_timesteps):
        # Count and rank tokens
        token_counts = Counter()
        for token_id, particle_list in position_data[t].items():
            token_counts[token_id] = len(particle_list)

        # Rank by frequency
        ranked_tokens = sorted(token_counts.items(), key=lambda x: (-x[1], x[0]))
        ranked_tokens = ranked_tokens[:20]  # Keep top 20

        rank_map = {}
        for rank, (token_id, count) in enumerate(ranked_tokens, 1):
            rank_map[token_id] = rank

            # Calculate average within-particle probability
            avg_prob = np.mean([
                p['within_particle_prob']
                for p in position_data[t][token_id]
            ])

            # Store metadata
            token_metadata[(t, rank)] = {
                'token_id': token_id,
                'frequency': count,
                'frequency_pct': count / len(particles),
                'avg_within_prob': avg_prob
            }

        position_ranks[t] = rank_map

    # Second pass: compute transition frequencies
    transition_counts = defaultdict(int)
    transition_totals = defaultdict(int)

    for particle in particles:
        for t in range(min(len(particle.tokens) - 1, max_timesteps - 1)):
            from_token = particle.tokens[t]
            to_token = particle.tokens[t + 1]

            from_rank = position_ranks[t].get(from_token, 21)
            to_rank = position_ranks[t + 1].get(to_token, 21)

            if from_rank <= 20 and to_rank <= 20:
                transition_counts[(t, from_rank, to_rank)] += 1
                transition_totals[(t, from_rank)] += 1

    # Compute transition frequencies
    transition_frequencies = {}
    for key, count in transition_counts.items():
        t, from_rank, _ = key
        total = transition_totals[(t, from_rank)]
        if total > 0:
            transition_frequencies[key] = count / len(particles)

    # Create DataFrame with trajectories
    data = {'timestep': list(range(max_timesteps))}

    for particle_idx, particle in enumerate(particles):
        trajectory = []
        for t in range(max_timesteps):
            if t < len(particle.tokens):
                token_id = particle.tokens[t]
                rank = position_ranks[t].get(token_id, 21)
                trajectory.append(rank)
            else:
                trajectory.append(np.nan)

        data[f'particle_{particle_idx}'] = trajectory

    df = pd.DataFrame(data)

    metadata = {
        'token_metadata': token_metadata,
        'transition_frequencies': transition_frequencies,
        'position_ranks': position_ranks,
        'n_particles': len(particles),
        'max_timesteps': max_timesteps
    }

    return df, metadata


def add_token_labels(
    ax: plt.Axes,
    metadata: Dict,
    tokenizer,
    colormap: str = 'RdYlGn',
    show_freq: bool = True
) -> None:
    """
    Add token labels with background coloring based on within-particle probability.

    Args:
        ax: Matplotlib axes
        metadata: Metadata with token information
        tokenizer: Tokenizer for decoding
        colormap: Same colormap as segments
        show_freq: Whether to show frequency percentages
    """
    from matplotlib.cm import get_cmap
    from matplotlib.colors import to_hex

    cmap = get_cmap(colormap)
    norm = Normalize(vmin=0, vmax=1)

    token_metadata = metadata.get('token_metadata', {})

    # Track used positions
    used_positions = set()

    for (timestep, rank), info in token_metadata.items():
        # Skip low frequency tokens
        if info['frequency_pct'] < 0.05 and rank > 10:
            continue

        # Decode token text
        try:
            token_text = tokenizer.decode([info['token_id']])
            token_text = token_text.strip().replace('\n', '↵').replace('\t', '→')
            if len(token_text) > 10:
                token_text = token_text[:8] + '..'
        except:
            token_text = f"<{info['token_id']}>"

        # Background color based on within-particle probability
        bg_color = cmap(norm(info['avg_within_prob']))

        # Adjust position to avoid overlap
        y_offset = 0
        key = (timestep, rank)
        for dy in [0, 0.3, -0.3, 0.6, -0.6]:
            test_key = (timestep, rank + dy)
            if test_key not in used_positions:
                y_offset = dy
                break

        # Add token label
        ax.text(
            timestep,
            rank + y_offset,
            token_text,
            fontsize=7 if info['frequency_pct'] > 0.1 else 6,
            ha='center',
            va='center',
            weight='bold' if info['frequency_pct'] > 0.5 else 'normal',
            bbox=dict(
                boxstyle='round,pad=0.2',
                facecolor=to_hex(bg_color),
                edgecolor='black',
                alpha=0.8,
                linewidth=0.5
            ),
            zorder=1000 + (20 - rank)  # Higher ranks on top
        )

        # Add frequency annotation
        if show_freq and info['frequency_pct'] > 0.15:
            ax.text(
                timestep,
                rank + y_offset - 0.35,
                f"{info['frequency_pct']:.0%}",
                fontsize=5,
                ha='center',
                va='top',
                color='darkgray',
                alpha=0.8,
                zorder=999
            )

        used_positions.add((timestep, rank + y_offset))