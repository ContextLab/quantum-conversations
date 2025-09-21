#!/usr/bin/env python3
"""Generate final demo bumplot visualizations with all fixes applied."""

import sys
sys.path.insert(0, '/Users/jmanning/quantum-conversations/code')

from quantum_conversations import ParticleFilter, TokenSequenceVisualizer, ModelManager
import matplotlib.pyplot as plt
from pathlib import Path

# Create output directory
output_dir = Path("bumplot_final_demos")
output_dir.mkdir(exist_ok=True)

print("Generating final demo visualizations...")
print("=" * 60)

model_manager = ModelManager()

# Test Case 1: Low temperature convergent
print("\n1. Low temperature convergent")
pf = ParticleFilter(
    model_name="EleutherAI/pythia-70m",
    n_particles=50,
    temperature=0.1,
    device="cpu",
    model_manager=model_manager,
    seed=42
)

particles = pf.generate("The future is", max_new_tokens=12)
viz = TokenSequenceVisualizer(tokenizer=pf.tokenizer)

fig = viz.visualize_bumplot(
    particles,
    output_path=str(output_dir / "final_low_temp.png"),
    max_vocab_display=10,
    show_tokens=True,
    colormap='RdYlGn',
    figsize=(16, 10)
)
plt.close(fig)
print(f"✓ Saved: {output_dir}/final_low_temp.png")

# Test Case 2: Medium temperature
print("\n2. Medium temperature balanced")
pf = ParticleFilter(
    model_name="EleutherAI/pythia-70m",
    n_particles=100,
    temperature=0.8,
    device="cpu",
    model_manager=model_manager,
    seed=42
)

particles = pf.generate("The future is", max_new_tokens=12)
viz = TokenSequenceVisualizer(tokenizer=pf.tokenizer)

fig = viz.visualize_bumplot(
    particles,
    output_path=str(output_dir / "final_medium_temp.png"),
    max_vocab_display=12,
    show_tokens=True,
    colormap='RdYlGn',
    figsize=(16, 10)
)
plt.close(fig)
print(f"✓ Saved: {output_dir}/final_medium_temp.png")

# Test Case 3: High temperature divergent
print("\n3. High temperature divergent")
pf = ParticleFilter(
    model_name="EleutherAI/pythia-70m",
    n_particles=150,
    temperature=1.5,
    device="cpu",
    model_manager=model_manager,
    seed=42
)

particles = pf.generate("The future is", max_new_tokens=12)
viz = TokenSequenceVisualizer(tokenizer=pf.tokenizer)

fig = viz.visualize_bumplot(
    particles,
    output_path=str(output_dir / "final_high_temp.png"),
    max_vocab_display=15,
    show_tokens=True,
    colormap='RdYlGn',
    figsize=(16, 10)
)
plt.close(fig)
print(f"✓ Saved: {output_dir}/final_high_temp.png")

# Test Case 4: Small example for curve visibility
print("\n4. Small example (10 particles)")
pf = ParticleFilter(
    model_name="EleutherAI/pythia-70m",
    n_particles=10,
    temperature=0.5,
    device="cpu",
    model_manager=model_manager,
    seed=42
)

particles = pf.generate("Hello world", max_new_tokens=8)
viz = TokenSequenceVisualizer(tokenizer=pf.tokenizer)

fig = viz.visualize_bumplot(
    particles,
    output_path=str(output_dir / "final_small_curves.png"),
    max_vocab_display=8,
    show_tokens=True,
    colormap='RdYlGn',
    figsize=(14, 8)
)
plt.close(fig)
print(f"✓ Saved: {output_dir}/final_small_curves.png")

print("\n" + "=" * 60)
print("IMPROVEMENTS IMPLEMENTED:")
print("=" * 60)
print("✓ Smooth curves without overshooting (quadratic interpolation)")
print("✓ No frequency labels on tokens")
print("✓ Narrow, short colorbar (1/3 height, only 0% and 100%)")
print("✓ Larger, readable token labels")
print("✓ All ranks with particles have labels")
print("✓ No figure title")
print("✓ 'Output position' as x-axis label")
print("✓ Most probable tokens at top (rank 1)")
print("✓ Thicker particle lines")
print("✓ Prompt tokens show 100% agreement")
print("\nVisualizations saved to: bumplot_final_demos/")