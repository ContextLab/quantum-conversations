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
    alpha: float = 0.2,  # Lower alpha for better transparency visibility
    linewidth: float = 2.5,
    curve_force: float = 0.3
) -> None:
    """
    Create custom bumplot with smooth spline curves and per-segment coloring.

    Args:
        df: DataFrame with particle trajectories
        metadata: Metadata including transition info
        ax: Matplotlib axes
        colormap: Colormap to use for probabilities
        alpha: Line transparency
        linewidth: Base line width (increased for visibility)
        curve_force: Intensity of curve bending (0-1)
    """
    from matplotlib.colors import to_hex
    from matplotlib.collections import LineCollection
    from scipy.interpolate import interp1d
    import matplotlib.pyplot as plt

    cmap = plt.colormaps[colormap]
    norm = Normalize(vmin=0, vmax=1)

    # Get transition frequencies from metadata
    transition_freqs = metadata.get('transition_frequencies', {})
    transition_probs = metadata.get('transition_probs', {})

    # Use transition_probs if available, fall back to transition_freqs
    if transition_probs:
        transition_freqs = transition_probs

    # Plot each particle trajectory
    particle_cols = [c for c in df.columns if c.startswith('particle_')]

    for col in particle_cols:
        trajectory = df[['timestep', col]].dropna()
        if len(trajectory) < 2:
            continue

        x = trajectory['timestep'].values
        y = trajectory[col].values

        # Create smooth curves without overshooting
        if len(x) > 2:
            try:
                # For each segment, create a smooth curve
                for i in range(len(x) - 1):
                    # Get transition frequency for coloring
                    from_rank = int(y[i])
                    to_rank = int(y[i+1])
                    timestep = int(x[i])

                    transition_key = (timestep, from_rank, to_rank)
                    freq = transition_freqs.get(transition_key, 0.0)
                    color = cmap(norm(freq))

                    # Create smooth S-curve between points
                    n_points = 50
                    t = np.linspace(0, 1, n_points)

                    # Use sigmoid for smooth transitions without overshooting
                    steepness = 5
                    transition = 1 / (1 + np.exp(-steepness * (t - 0.5) * 2))

                    x_segment = x[i] + t * (x[i+1] - x[i])
                    y_segment = y[i] + transition * (y[i+1] - y[i])

                    # Add overlap to prevent gaps
                    if i < len(x) - 2:
                        # Extend slightly into next segment
                        overlap_points = 2
                        t_overlap = np.linspace(0, 0.05, overlap_points)
                        x_overlap = x[i+1] + t_overlap * (x[i+2] - x[i+1])
                        y_overlap = y[i+1] + t_overlap * 0  # Stay at same rank initially
                        x_segment = np.concatenate([x_segment, x_overlap])
                        y_segment = np.concatenate([y_segment, y_overlap])

                    # Plot this segment with explicit alpha
                    ax.plot(x_segment, y_segment, color=color, alpha=alpha,
                           linewidth=linewidth, solid_capstyle='round',
                           solid_joinstyle='round', zorder=100, rasterized=False)  # Below labels

            except Exception as e:
                # Fallback to linear segments
                for i in range(len(x) - 1):
                    from_rank = int(y[i])
                    to_rank = int(y[i+1])
                    timestep = int(x[i])

                    transition_key = (timestep, from_rank, to_rank)
                    freq = transition_freqs.get(transition_key, 0.0)
                    color = cmap(norm(freq))

                    ax.plot([x[i], x[i+1]], [y[i], y[i+1]],
                           color=color, alpha=alpha, linewidth=linewidth)
        else:
            # For trajectories with only 2 points, draw straight line
            for i in range(len(x) - 1):
                from_rank = int(y[i])
                to_rank = int(y[i+1])
                timestep = int(x[i])

                transition_key = (timestep, from_rank, to_rank)
                freq = transition_freqs.get(transition_key, 0.0)
                color = cmap(norm(freq))

                ax.plot([x[i], x[i+1]], [y[i], y[i+1]],
                       color=color, alpha=alpha, linewidth=linewidth)


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
    show_freq: bool = False,  # Changed to False - no frequency labels
    min_freq_threshold: float = 0.05,
    max_labels_per_timestep: int = 15
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

            # Show all tokens that have a rank (they have particles there)
            # This ensures no "empty" ranks

            # Decode token text (use cache if available)
            token_text = metadata.get('token_to_text', {}).get(info['token_id'])
            if not token_text:
                try:
                    token_text = tokenizer.decode([info['token_id']])
                    token_text = token_text.strip()
                    # Better special character handling
                    token_text = token_text.replace('\n', '↵').replace('\t', '→')
                    token_text = token_text.replace('\r', '↲')
                    # Increased max length, smart truncation
                    if len(token_text) > 15:
                        # Try to break at word boundary
                        if ' ' in token_text[:15]:
                            token_text = token_text[:token_text[:15].rfind(' ')] + '..'
                        else:
                            token_text = token_text[:13] + '..'
                    # Handle empty tokens
                    if not token_text or token_text.isspace():
                        token_text = '[space]'  # Text representation for space
                except Exception as e:
                    token_text = f"<{info['token_id']}>"

            # Calculate colors
            # Background: within-particle probability
            bg_color = cmap(norm(info['avg_within_prob']))
            # Opaque background for readability
            bg_alpha = 1.0

            # Determine text color based on background brightness
            # Convert to RGB for brightness calculation
            import matplotlib.colors as mcolors
            rgb = mcolors.to_rgb(bg_color)
            # Calculate perceived brightness (weighted average)
            brightness = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            # Use white text on dark backgrounds, black on light
            text_color = 'white' if brightness < 0.5 else 'black'

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

            # Adaptive font size based on token length
            if len(token_text) > 10:
                fontsize = 10  # Smaller for long tokens
            else:
                fontsize = 12  # Standard size
            weight = 'normal'  # Consistent weight

            # Add token label with opaque background
            text_obj = ax.text(
                test_x,
                test_y,
                token_text,
                fontsize=fontsize,
                ha='center',
                va='center',
                weight=weight,
                color=text_color,  # Smart color selection
                bbox=dict(
                    boxstyle='round,pad=0.2',
                    facecolor=to_hex(bg_color),
                    edgecolor='none',  # No border for cleaner look
                    alpha=bg_alpha,  # Fully opaque
                    linewidth=0
                ),
                zorder=10000  # Above everything else
            )

            # Add to placed boxes
            placed_boxes.append(bbox)
            labels_added += 1

            # Removed frequency annotations per requirements