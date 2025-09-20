"""
Demo with 1000 particles and 20 token sequences as requested.
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

print("=== Quantum Conversations Demo: 1000 Particles, 20 Steps ===\n")

# Initialize with 1000 particles as requested
print("Initializing particle filter with 1000 particles...")
pf = ParticleFilter(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    n_particles=1000,
    temperature=1.0,
    top_k=100,
    top_p=0.95,
    device="cpu"
)
print("Model loaded!")

# Visualizer settings for 1000 particles
viz = TokenSequenceVisualizer(
    tokenizer=pf.tokenizer,
    figsize=(20, 12),
    alpha=0.01,     # Very low alpha for 1000 particles
    line_width=0.1  # Very thin lines
)

# Output directory
output_dir = "../data/derivatives/particle_visualizations/demo_1000_final"
os.makedirs(output_dir, exist_ok=True)

# Generate for one prompt
prompt = "The most surprising thing was "
print(f"\nGenerating 1000 particles for: '{prompt}'")
print("Each particle will generate 20 tokens...")
print("This will take several minutes...\n")

# Generate particles
particles = pf.generate(prompt, max_new_tokens=20)

# Analysis
print("Analyzing results...")
sequences = pf.get_token_sequences()
log_probs = [lp for _, lp in sequences]
best_idx = max(range(len(log_probs)), key=lambda i: log_probs[i])
best_text = sequences[best_idx][0]

unique_seqs = len(set(tuple(p.tokens) for p in particles))
print(f"\nMost probable sequence: {best_text}")
print(f"Unique sequences: {unique_seqs}/1000 ({unique_seqs/10:.1f}%)")

# Create Sankey diagram
print("\nCreating Sankey diagram with 1000 paths...")
fig = viz.visualize(
    particles=particles,
    prompt=prompt,
    output_path=os.path.join(output_dir, "sankey_1000_particles.png"),
    title="Token Generation Paths: 1000 Particles Exploring Language Space"
)
plt.close(fig)
print("✓ Saved sankey_1000_particles.png")

# Create heatmap
print("\nCreating probability heatmap...")
fig = viz.visualize_probability_heatmap(
    particles=particles,
    prompt=prompt,
    output_path=os.path.join(output_dir, "heatmap_1000_particles.png"),
    vocab_size=32000
)
plt.close(fig)
print("✓ Saved heatmap_1000_particles.png")

print(f"\nVisualizations saved to: {os.path.abspath(output_dir)}")
print("\nWith 1000 particles, the visualizations show:")
print("- Sankey: A cloud of 1000 black paths (alpha=0.01) with the most probable in red")
print("- Heatmap: The full probability distribution across the 32K token vocabulary")
print("- The 'quantum' nature of language generation through multiple possible paths")