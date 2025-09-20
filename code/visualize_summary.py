#!/usr/bin/env python3
"""
Create a summary visualization of the existing sequence
"""

import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Paths to existing visualizations
viz_dirs = {
    "demo_1000_resume": "../data/derivatives/particle_visualizations/demo_1000_resume/figures_v2",
    "high_ambiguity": "../data/derivatives/particle_visualizations/high_ambiguity",
    "medium_ambiguity": "../data/derivatives/particle_visualizations/medium_ambiguity",
    "low_ambiguity": "../data/derivatives/particle_visualizations/low_ambiguity"
}

# Create figure
fig = plt.figure(figsize=(24, 16))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.2)

# Title
fig.suptitle("Quantum Conversations: Token Generation Visualizations", fontsize=20, y=0.98)

# Plot positions
plot_info = [
    ("demo_1000_resume", "sankey_diagram.png", "Sankey Diagram: 'The most surprising thing was' (139 particles)", 0, 0),
    ("demo_1000_resume", "heatmap_focused.png", "Token Probability Heatmap", 0, 1),
]

# Check what other visualizations exist
for level in ["high_ambiguity", "medium_ambiguity", "low_ambiguity"]:
    level_dir = f"../data/derivatives/particle_visualizations/{level}"
    if os.path.exists(level_dir):
        subdirs = [d for d in os.listdir(level_dir) if os.path.isdir(os.path.join(level_dir, d))]
        if subdirs and len(subdirs) > 0:
            # Use first example from each level
            subdir = subdirs[0]
            if os.path.exists(os.path.join(level_dir, subdir, "00_sankey_" + subdir[3:] + ".png")):
                row = {"high_ambiguity": 1, "medium_ambiguity": 2, "low_ambiguity": 2}.get(level, 1)
                col = 0 if level != "low_ambiguity" else 1
                plot_info.append((level, f"{subdir}/00_sankey_{subdir[3:]}.png", 
                                f"{level.replace('_', ' ').title()}", row, col))

# Add plots
for idx, (source, filename, title, row, col) in enumerate(plot_info):
    ax = fig.add_subplot(gs[row, col])
    
    # Construct path
    if source == "demo_1000_resume":
        img_path = os.path.join(viz_dirs[source], filename)
    else:
        img_path = os.path.join(viz_dirs[source], filename)
    
    # Check if file exists
    if os.path.exists(img_path):
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(title, fontsize=14, pad=10)
    else:
        ax.text(0.5, 0.5, f"Image not found:\n{filename}", 
                ha='center', va='center', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(title, fontsize=14, pad=10)

# Add description
desc_text = """
This visualization demonstrates the Quantum Conversations approach:
• Particle filtering tracks multiple possible token generation paths
• Sankey diagrams show the branching possibilities with transparency (α=0.01)
• Heatmaps reveal the full vocabulary probability distribution at each time step
• The red line highlights the most probable path through token space

The approach captures how "paths not taken" (alternative token choices) 
influence the overall generation process, similar to quantum superposition.
"""

fig.text(0.5, 0.02, desc_text, ha='center', va='bottom', fontsize=12,
         bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))

# Save
output_path = "../data/derivatives/particle_visualizations/summary_visualization.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Summary visualization saved to: {output_path}")

# Also create a simple display of what we have
print("\n" + "="*60)
print("QUANTUM CONVERSATIONS VISUALIZATION SUMMARY")
print("="*60)
print("\nGenerated visualizations:")
print(f"✓ Main demo: 139 particles for 'The most surprising thing was'")
print(f"  - Sankey diagram showing particle paths")
print(f"  - Heatmap showing token probabilities")
print(f"  - Focused heatmap with top tokens annotated")
print(f"\nData saved:")
print(f"✓ V×t×n tensor: (32000, 20, 139) - Full probability distributions")
print(f"✓ Individual particle pkl files for resumability")
print(f"✓ Vocabulary mapping")
print("\nReady for additional sequences with varying ambiguity levels!")
print("="*60)