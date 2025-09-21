#!/usr/bin/env python3
"""Generate final demo showing all bumplot improvements."""

import sys
sys.path.insert(0, '/Users/jmanning/quantum-conversations/code')

from quantum_conversations import ParticleFilter, TokenSequenceVisualizer, ModelManager
import matplotlib.pyplot as plt
from pathlib import Path

# Create output directory
output_dir = Path("bumplot_final")
output_dir.mkdir(exist_ok=True)

print("Generating final demo visualizations with all improvements...")
print("=" * 60)

model_manager = ModelManager()

# Test Case 1: Small example to see all improvements clearly
print("\n1. Small example (20 particles)")
pf = ParticleFilter(
    model_name="EleutherAI/pythia-70m",
    n_particles=20,
    temperature=0.7,
    device="cpu",
    model_manager=model_manager,
    seed=42
)

particles = pf.generate("The future is", max_new_tokens=10)
viz = TokenSequenceVisualizer(tokenizer=pf.tokenizer)

fig = viz.visualize_bumplot(
    particles,
    output_path=str(output_dir / "final_demo_small.png"),
    max_vocab_display=8,
    show_tokens=True,
    colormap='RdYlGn',
    figsize=(16, 10)
)
plt.close(fig)
print(f"✓ Saved: {output_dir}/final_demo_small.png")

# Test Case 2: Medium complexity to show transparency
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
    output_path=str(output_dir / "final_demo_medium.png"),
    max_vocab_display=10,
    show_tokens=True,
    colormap='RdYlGn',
    figsize=(16, 10)
)
plt.close(fig)
print(f"✓ Saved: {output_dir}/final_demo_medium.png")

# Test Case 3: High divergence
print("\n3. High divergence (100 particles)")
pf = ParticleFilter(
    model_name="EleutherAI/pythia-70m",
    n_particles=100,
    temperature=1.2,
    device="cpu",
    model_manager=model_manager,
    seed=42
)

particles = pf.generate("Once upon a", max_new_tokens=12)
viz = TokenSequenceVisualizer(tokenizer=pf.tokenizer)

fig = viz.visualize_bumplot(
    particles,
    output_path=str(output_dir / "final_demo_divergent.png"),
    max_vocab_display=12,
    show_tokens=True,
    colormap='RdYlGn',
    figsize=(18, 10)
)
plt.close(fig)
print(f"✓ Saved: {output_dir}/final_demo_divergent.png")

print("\n" + "=" * 60)
print("ALL IMPROVEMENTS IMPLEMENTED:")
print("=" * 60)
print("✓ No overshooting (sigmoid transitions)")
print("✓ No gaps (overlapping segments)")
print("✓ No out-of-range curves (clamped to max_vocab_display)")
print("✓ Clear transparency (alpha=0.1)")
print("✓ Uniform font size (12pt, 10pt for long tokens)")
print("✓ Opaque token backgrounds")
print("✓ Smart text colors (white/black based on background)")
print("✓ Large axis labels (18pt)")
print("✓ Large tick labels (14pt)")
print("✓ Colorbar at x=0.88 with 0%/100% labels")
print("✓ Proper y-axis padding (0.3 to max+0.8)")
print("✓ Max token length 15 chars")
print("\nFinal visualizations saved to: bumplot_final/")