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
    linewidth: float = 1.5,
    curve_force: float = 0.5
) -> None:
    """
    Create custom bumplot with smooth spline curves and per-segment coloring.

    Args:
        df: DataFrame with particle trajectories
        metadata: Metadata including transition info
        ax: Matplotlib axes
        colormap: Colormap to use for probabilities
        alpha: Line transparency
        linewidth: Line width
        curve_force: Intensity of curve bending (0-1)
    """
    from matplotlib.colors import to_hex
    from matplotlib.collections import LineCollection
    from scipy.interpolate import make_interp_spline
    import matplotlib.pyplot as plt

    cmap = plt.colormaps[colormap]
    norm = Normalize(vmin=0, vmax=1)

    # Get transition frequencies from metadata
    transition_freqs = metadata.get('transition_frequencies', {})
    transition_probs = metadata.get('transition_probs', {})

    # Use transition_probs if available, fall back to transition_freqs
    if transition_probs:
        transition_freqs = transition_probs

    # Plot each particle trajectory with smooth curves
    particle_cols = [c for c in df.columns if c.startswith('particle_')]

    # Group particles by their trajectories to ensure identical paths converge
    trajectory_groups = defaultdict(list)
    for col in particle_cols:
        trajectory = df[['timestep', col]].dropna()
        if len(trajectory) >= 2:
            # Create a hashable key from the trajectory
            traj_key = tuple(zip(trajectory['timestep'].values, trajectory[col].values))
            trajectory_groups[traj_key].append(col)

    # Plot each unique trajectory
    for traj_key, particle_list in trajectory_groups.items():
        # Convert back to arrays
        points = np.array(traj_key)
        x = points[:, 0]
        y = points[:, 1]

        if len(x) < 2:
            continue

        # Create smooth segments between each pair of points
        for i in range(len(x) - 1):
            x_seg = np.array([x[i], x[i+1]])
            y_seg = np.array([y[i], y[i+1]])

            # Get transition frequency for coloring
            from_rank = int(y[i])
            to_rank = int(y[i+1])
            timestep = int(x[i])

            # Look up transition frequency
            transition_key = (timestep, from_rank, to_rank)
            freq = transition_freqs.get(transition_key, 0.0)

            # Determine color and width based on frequency
            color = cmap(norm(freq))
            segment_width = linewidth * (0.3 + freq * 0.7)  # Scale by frequency
            segment_alpha = alpha * (0.5 + freq * 0.5)  # More opaque for common transitions

            # Create smooth curve between points
            rank_change = abs(y_seg[1] - y_seg[0])

            if rank_change > 0.1:  # Only curve if there's significant movement
                try:
                    # Use spline interpolation for smooth S-curves
                    # Add control points for better curves
                    control_offset = curve_force * 0.4  # How far to push control points

                    # Create control points for cubic spline
                    if rank_change > 2:  # Strong S-curve for large transitions
                        # Add intermediate control points
                        x_control = np.array([
                            x_seg[0],
                            x_seg[0] + (x_seg[1] - x_seg[0]) * 0.3,
                            x_seg[0] + (x_seg[1] - x_seg[0]) * 0.7,
                            x_seg[1]
                        ])
                        y_control = np.array([
                            y_seg[0],
                            y_seg[0],  # Hold at starting rank
                            y_seg[1],  # Jump to ending rank
                            y_seg[1]
                        ])
                    else:  # Gentle curve for small transitions
                        x_control = np.array([x_seg[0], x_seg[0] + (x_seg[1]-x_seg[0])*0.5, x_seg[1]])
                        y_control = np.array([y_seg[0], (y_seg[0] + y_seg[1])*0.5, y_seg[1]])

                    # Create spline with appropriate degree
                    k = min(3, len(x_control) - 1)  # Cubic or less
                    if k >= 1:
                        spl = make_interp_spline(x_control, y_control, k=k)

                        # Generate smooth points
                        x_smooth = np.linspace(x_seg[0], x_seg[1], 50)
                        y_smooth = spl(x_smooth)

                        # Plot the smooth curve
                        ax.plot(x_smooth, y_smooth, color=color, alpha=segment_alpha,
                               linewidth=segment_width, solid_capstyle='round',
                               solid_joinstyle='round')
                    else:
                        # Fallback to straight line
                        ax.plot(x_seg, y_seg, color=color, alpha=segment_alpha,
                               linewidth=segment_width)

                except Exception as e:
                    logger.debug(f"Spline interpolation failed: {e}, using straight line")
                    # Fallback to straight line
                    ax.plot(x_seg, y_seg, color=color, alpha=segment_alpha,
                           linewidth=segment_width)
            else:
                # Straight line for no/minimal rank change
                ax.plot(x_seg, y_seg, color=color, alpha=segment_alpha,
                       linewidth=segment_width)


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
    show_freq: bool = True,
    min_freq_threshold: float = 0.05,
    max_labels_per_timestep: int = 10
) -> None:
    """
    Add token labels with smart positioning and background coloring.
    Implements collision detection and priority-based label placement.

    Args:
        ax: Matplotlib axes
        metadata: Metadata with token information
        tokenizer: Tokenizer for decoding
        colormap: Same colormap as segments
        show_freq: Whether to show frequency percentages
        min_freq_threshold: Minimum frequency to show label
        max_labels_per_timestep: Maximum labels per timestep
    """
    from matplotlib.colors import to_hex
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.transforms as transforms
    import matplotlib.pyplot as plt

    cmap = plt.colormaps[colormap]
    norm = Normalize(vmin=0, vmax=1)

    token_metadata = metadata.get('token_metadata', {})
    if not token_metadata:
        return

    # Get axis limits for boundary checking
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Group tokens by timestep for better layout
    timestep_tokens = defaultdict(list)
    for (timestep, rank), info in token_metadata.items():
        timestep_tokens[timestep].append((rank, info))

    # Sort each timestep by frequency (highest first)
    for timestep in timestep_tokens:
        timestep_tokens[timestep].sort(key=lambda x: -x[1]['frequency_pct'])

    # Track bounding boxes for collision detection
    placed_boxes = []

    # Process each timestep
    for timestep in sorted(timestep_tokens.keys()):
        labels_added = 0

        for rank, info in timestep_tokens[timestep]:
            # Apply thresholds
            if labels_added >= max_labels_per_timestep:
                break

            # Always show high-frequency tokens (>20%)
            # Show medium frequency (5-20%) if space available
            # Skip very low frequency (<5%) unless top 5
            if info['frequency_pct'] < 0.2:
                if info['frequency_pct'] < min_freq_threshold and labels_added >= 5:
                    continue
                if rank > 10 and info['frequency_pct'] < 0.1:
                    continue

            # Decode token text (use cache if available)
            token_text = metadata.get('token_to_text', {}).get(info['token_id'])
            if not token_text:
                try:
                    token_text = tokenizer.decode([info['token_id']])
                    token_text = token_text.strip()
                    # Better special character handling
                    token_text = token_text.replace('\n', '↵').replace('\t', '→')
                    token_text = token_text.replace('\r', '↲')
                    # Smart truncation
                    if len(token_text) > 10:
                        # Try to break at word boundary
                        if ' ' in token_text[:10]:
                            token_text = token_text[:token_text[:10].rfind(' ')] + '..'
                        else:
                            token_text = token_text[:8] + '..'
                    # Handle empty tokens
                    if not token_text or token_text.isspace():
                        token_text = '[space]'  # Text representation for space
                except Exception as e:
                    token_text = f"<{info['token_id']}>"

            # Calculate colors
            # Background: within-particle probability
            bg_color = cmap(norm(info['avg_within_prob']))
            # Make more transparent for lower frequency
            bg_alpha = 0.3 + info['frequency_pct'] * 0.5

            # Find non-overlapping position
            base_x = timestep
            base_y = rank

            # Try different positions with smart offset
            position_found = False
            for attempt in range(5):
                if attempt == 0:
                    x_offset = 0
                    y_offset = 0
                elif attempt == 1:
                    x_offset = 0.15 if timestep < xlim[1] - 0.5 else -0.15
                    y_offset = 0
                elif attempt == 2:
                    x_offset = 0
                    y_offset = 0.3 if rank < ylim[1] - 1 else -0.3
                elif attempt == 3:
                    x_offset = -0.15 if timestep > xlim[0] + 0.5 else 0.15
                    y_offset = 0
                else:
                    x_offset = 0
                    y_offset = -0.3 if rank > ylim[0] + 1 else 0.3

                test_x = base_x + x_offset
                test_y = base_y + y_offset

                # Estimate text bbox (rough approximation)
                text_width = len(token_text) * 0.08  # Approximate width
                text_height = 0.3  # Approximate height
                bbox = (test_x - text_width/2, test_y - text_height/2,
                       test_x + text_width/2, test_y + text_height/2)

                # Check for collisions
                collision = False
                for existing_box in placed_boxes:
                    if (bbox[0] < existing_box[2] and bbox[2] > existing_box[0] and
                        bbox[1] < existing_box[3] and bbox[3] > existing_box[1]):
                        collision = True
                        break

                if not collision:
                    position_found = True
                    break

            if not position_found:
                # Skip this label if no position found
                continue

            # Determine font size based on importance
            if info['frequency_pct'] > 0.5:
                fontsize = 8
                weight = 'bold'
            elif info['frequency_pct'] > 0.2:
                fontsize = 7
                weight = 'semibold'
            else:
                fontsize = 6
                weight = 'normal'

            # Add token label
            text_obj = ax.text(
                test_x,
                test_y,
                token_text,
                fontsize=fontsize,
                ha='center',
                va='center',
                weight=weight,
                bbox=dict(
                    boxstyle='round,pad=0.15',
                    facecolor=to_hex(bg_color),
                    edgecolor='#333333',
                    alpha=bg_alpha,
                    linewidth=0.5 if info['frequency_pct'] < 0.3 else 0.8
                ),
                zorder=2000 - rank  # Higher frequency on top
            )

            # Add to placed boxes
            placed_boxes.append(bbox)
            labels_added += 1

            # Add frequency annotation for significant tokens
            if show_freq and info['frequency_pct'] >= 0.2:
                freq_text = f"{info['frequency_pct']:.0%}"
                ax.text(
                    test_x,
                    test_y - 0.25,
                    freq_text,
                    fontsize=5,
                    ha='center',
                    va='top',
                    color='#555555',
                    alpha=0.8,
                    weight='bold',
                    zorder=1999
                )