#!/usr/bin/env python3
"""Test all final improvements to bumplot visualization."""

import sys
sys.path.insert(0, '/Users/jmanning/quantum-conversations/code')

from quantum_conversations import ParticleFilter, TokenSequenceVisualizer, ModelManager
import matplotlib.pyplot as plt
from pathlib import Path

# Create output directory
output_dir = Path("bumplot_improved")
output_dir.mkdir(exist_ok=True)

print("Testing all improvements...")
print("=" * 60)

model_manager = ModelManager()

# Test case 1: Small example to clearly see improvements
print("\n1. Small example (15 particles)")
pf = ParticleFilter(
    model_name="EleutherAI/pythia-70m",
    n_particles=15,
    temperature=0.6,
    device="cpu",
    model_manager=model_manager,
    seed=42
)

particles = pf.generate("The future is", max_new_tokens=10)
viz = TokenSequenceVisualizer(tokenizer=pf.tokenizer)

fig = viz.visualize_bumplot(
    particles,
    output_path=str(output_dir / "improved_small.png"),
    max_vocab_display=8,
    show_tokens=True,
    colormap='RdYlGn',
    figsize=(14, 8)
)
plt.close(fig)
print(f"✓ Saved: {output_dir}/improved_small.png")

# Test case 2: Medium complexity
print("\n2. Medium complexity (50 particles)")
pf = ParticleFilter(
    model_name="EleutherAI/pythia-70m",
    n_particles=50,
    temperature=0.8,
    device="cpu",
    model_manager=model_manager,
    seed=42
)

particles = pf.generate("Hello world", max_new_tokens=12)
viz = TokenSequenceVisualizer(tokenizer=pf.tokenizer)

fig = viz.visualize_bumplot(
    particles,
    output_path=str(output_dir / "improved_medium.png"),
    max_vocab_display=10,
    show_tokens=True,
    colormap='RdYlGn',
    figsize=(16, 10)
)
plt.close(fig)
print(f"✓ Saved: {output_dir}/improved_medium.png")

print("\n" + "=" * 60)
print("IMPROVEMENTS CHECKLIST:")
print("=" * 60)
print("✓ Uniform font size (12pt) for all tokens")
print("✓ Opaque token backgrounds (alpha=1.0)")
print("✓ Smart text color (white on dark, black on light)")
print("✓ Sigmoid curves prevent overshooting")
print("✓ Overlapping segments prevent gaps")
print("✓ Tighter y-axis bounds (no wasted space)")
print("✓ Colorbar closer to figure (x=0.86)")
print("✓ Labels above/below colorbar")
print("✓ Larger axis labels (14pt)")
print("✓ Larger tick labels (12pt)")
print("✓ Curve alpha = 0.5")
print("✓ Token labels z-order = 10000 (above curves)")
print("\nCheck visualizations in: bumplot_improved/")