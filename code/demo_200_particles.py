"""
Demo with 200 particles (scaled down from 1000 for execution time).
Shows the concept with 20 token sequences.
"""

import os
import sys
sys.path.append('.')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from quantum_conversations import ParticleFilter, TokenSequenceVisualizer

print("=== Quantum Conversations Demo: 200 Particles, 20 Steps ===")
print("(In practice, use 1000-10000 particles)\n")

# Initialize with 200 particles
pf = ParticleFilter(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    n_particles=200,
    temperature=1.0,
    top_k=100,
    top_p=0.95,
    device="cpu"
)

# Visualizer settings
viz = TokenSequenceVisualizer(
    tokenizer=pf.tokenizer,
    figsize=(20, 12),
    alpha=0.02,     # Slightly higher alpha since fewer particles
    line_width=0.2
)

# Output directory
output_dir = "../data/derivatives/particle_visualizations/demo_200"
os.makedirs(output_dir, exist_ok=True)

# Single example to demonstrate
prompt = "The most surprising thing was "
print(f"Generating 200 particles for: '{prompt}'")
print("Each particle will generate 20 tokens...\n")

# Generate
particles = pf.generate(prompt, max_new_tokens=20)

# Analysis
sequences = pf.get_token_sequences()
log_probs = [lp for _, lp in sequences]
best_idx = max(range(len(log_probs)), key=lambda i: log_probs[i])
best_text = sequences[best_idx][0]

unique_seqs = len(set(tuple(p.tokens) for p in particles))
print(f"Most probable sequence: {best_text}")
print(f"Unique sequences: {unique_seqs}/200 ({unique_seqs/2:.1f}%)")

# Sankey diagram
print("\nCreating Sankey diagram...")
fig = viz.visualize(
    particles=particles,
    prompt=prompt,
    output_path=os.path.join(output_dir, "sankey_200_particles.png"),
    title="Token Generation Paths: 200 Particles Exploring Possibilities"
)
plt.close(fig)
print("✓ Saved sankey_200_particles.png")

# Heatmap
print("Creating probability heatmap...")
fig = viz.visualize_probability_heatmap(
    particles=particles,
    prompt=prompt,
    output_path=os.path.join(output_dir, "heatmap_200_particles.png"),
    vocab_size=32000
)
plt.close(fig)
print("✓ Saved heatmap_200_particles.png")

print(f"\nVisualizations saved to: {os.path.abspath(output_dir)}")
print("\nThe visualizations show:")
print("- Sankey: 200 black paths (alpha=0.02) with the most probable in red")
print("- Heatmap: Full token probability distribution across all 32K tokens")