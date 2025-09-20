"""
Run demo with 1000 particles and 20 steps to generate figures.
"""

import os
import sys
sys.path.append('.')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from quantum_conversations import ParticleFilter, TokenSequenceVisualizer

print("=== Quantum Conversations Demo: 1000 Particles, 20 Steps ===\n")

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize particle filter with 1000 particles
print("\nInitializing particle filter with 1,000 particles...")
pf = ParticleFilter(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    n_particles=1000,
    temperature=1.0,
    top_k=100,
    top_p=0.95,
    device=device
)

print("Model loaded successfully!")

# Initialize visualizer
viz = TokenSequenceVisualizer(
    tokenizer=pf.tokenizer,
    figsize=(20, 12),
    alpha=0.01,      # Very low alpha for 1000 particles
    line_width=0.1   # Very thin lines
)

# Create output directory
output_dir = "../data/derivatives/particle_visualizations/demo_1000"
os.makedirs(output_dir, exist_ok=True)

# Test with three prompts of different ambiguity
test_prompts = [
    ("high_ambiguity", "The most surprising thing was "),
    ("medium_ambiguity", "The recipe called for "),
    ("low_ambiguity", "Two plus two equals ")
]

print("\nGenerating visualizations...\n")

for label, prompt in test_prompts:
    print(f"Processing {label}: '{prompt}'")
    print("  Generating 1000 particle paths (20 tokens each)...")
    
    # Generate particles
    particles = pf.generate(prompt, max_new_tokens=20)
    
    # Find most probable sequence
    sequences = pf.get_token_sequences()
    log_probs = [lp for _, lp in sequences]
    most_probable_idx = max(range(len(log_probs)), key=lambda i: log_probs[i])
    most_probable_text = sequences[most_probable_idx][0]
    print(f"  Most probable: {most_probable_text[:80]}...")
    
    # Calculate diversity
    unique_sequences = len(set(tuple(p.tokens) for p in particles))
    print(f"  Unique sequences: {unique_sequences}/1000 ({unique_sequences/10:.1f}%)")
    
    # Create Sankey diagram
    print("  Creating Sankey diagram...")
    fig = viz.visualize(
        particles=particles,
        prompt=prompt,
        output_path=os.path.join(output_dir, f"{label}_sankey.png"),
        title=f"Token Generation Paths ({label.replace('_', ' ').title()})"
    )
    plt.close(fig)
    print(f"  ✓ Saved {label}_sankey.png")
    
    # Create heatmap
    print("  Creating probability heatmap...")
    fig = viz.visualize_probability_heatmap(
        particles=particles,
        prompt=prompt,
        output_path=os.path.join(output_dir, f"{label}_heatmap.png"),
        vocab_size=32000
    )
    plt.close(fig)
    print(f"  ✓ Saved {label}_heatmap.png")
    
    print()

print(f"All visualizations saved to: {os.path.abspath(output_dir)}")
print("\nWith 1000 particles, you can see:")
print("- High ambiguity prompts create diverse, spreading paths")
print("- Low ambiguity prompts converge quickly to similar outputs")
print("- The 'quantum' nature of language generation is visible in the branching paths")