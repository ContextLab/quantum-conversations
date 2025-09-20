#!/usr/bin/env python3
"""
Create visualizations from existing pkl files
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from quantum_conversations.visualizer import TokenSequenceVisualizer
from quantum_conversations.particle_filter import Particle

# Load the tensor and vocabulary
tensor_path = "../data/derivatives/particle_visualizations/demo_1000_resume/token_probabilities_tensor.pkl"
vocab_path = "../data/derivatives/particle_visualizations/demo_1000_resume/vocabulary.pkl"

print("Loading tensor data...")
with open(tensor_path, 'rb') as f:
    tensor_data = pickle.load(f)

print("Loading vocabulary...")
with open(vocab_path, 'rb') as f:
    vocab_data = pickle.load(f)

# Extract data
tensor = tensor_data['tensor']  # Shape: (V, t, n)
prompt = tensor_data['prompt']
vocabulary = vocab_data  # vocab_data is already the list

print(f"\nTensor shape: {tensor.shape}")
print(f"Prompt: '{prompt}'")
print(f"Vocabulary size: {len(vocabulary)}")

# Load tokenizer for visualization
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"\nLoading tokenizer from {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create visualizer with appropriate settings
visualizer = TokenSequenceVisualizer(
    tokenizer=tokenizer,
    figsize=(20, 12),
    max_tokens_display=50,
    alpha=0.01,  # Very thin lines
    line_width=0.5
)

# Find the most probable path
print("\nFinding most probable path...")
n_particles = tensor.shape[2]
n_steps = tensor.shape[1]

# For each particle, compute total log probability
particle_log_probs = []
particle_tokens = []

for particle_id in range(n_particles):
    # Get the token sequence for this particle by finding max prob at each step
    tokens = []
    total_log_prob = 0.0
    
    for t in range(n_steps):
        # Get probability distribution for this particle at this time
        probs = tensor[:, t, particle_id]
        
        # Find the token that was actually selected (highest probability)
        selected_token_id = np.argmax(probs)
        tokens.append(selected_token_id)
        
        # Add log probability
        if probs[selected_token_id] > 0:
            total_log_prob += np.log(probs[selected_token_id])
    
    particle_tokens.append(tokens)
    particle_log_probs.append(total_log_prob)

# Find best particle
best_particle_idx = np.argmax(particle_log_probs)
print(f"Best particle: {best_particle_idx} with log prob: {particle_log_probs[best_particle_idx]:.4f}")

# Convert token IDs to strings for best path
best_tokens = particle_tokens[best_particle_idx]
best_text = tokenizer.decode(best_tokens, skip_special_tokens=True)
print(f"Best path text: '{prompt}{best_text}'")

# Prepare data for visualization
# Extract token sequences and probabilities for all particles
all_sequences = []
all_probs = []

for particle_id in range(n_particles):
    sequence = particle_tokens[particle_id]
    all_sequences.append(sequence)
    
    # Extract probability sequence
    prob_sequence = []
    for t, token_id in enumerate(sequence):
        prob_sequence.append(tensor[token_id, t, particle_id])
    all_probs.append(prob_sequence)

# Create output directory
output_dir = "../data/derivatives/particle_visualizations/demo_1000_resume/figures"
os.makedirs(output_dir, exist_ok=True)

# Create Particle objects for visualization
print("\nCreating particle objects...")
particles = []
for i in range(n_particles):
    particle = Particle(
        tokens=particle_tokens[i],
        log_prob=particle_log_probs[i],
        token_probs_history=[]  # We don't have this data, but it's not used in visualization
    )
    particles.append(particle)

# Create Sankey diagram using the visualize method
print("\nCreating Sankey diagram...")
fig = visualizer.visualize(
    particles=particles,
    prompt=prompt,
    output_path=None,
    title="Token Generation Paths",
    highlight_most_probable=True
)
sankey_path = os.path.join(output_dir, "sankey_diagram.png")
fig.savefig(sankey_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved Sankey diagram to {sankey_path}")

# Create heatmap separately
print("\nCreating heatmap...")
# For heatmap, we'll show the average probability distribution across all particles
avg_probs = np.mean(tensor, axis=2)  # Average across particles: (V, t)

# Create a custom heatmap since the visualizer doesn't have this method
fig, ax = plt.subplots(figsize=(20, 12))

# Select top tokens by average probability at each time step
n_top_tokens = 50
top_token_indices = []

for t in range(n_steps):
    # Get top tokens at this time step
    top_indices = np.argsort(avg_probs[:, t])[-n_top_tokens:][::-1]
    top_token_indices.extend(top_indices)

# Get unique token indices
unique_indices = sorted(list(set(top_token_indices)))[:n_top_tokens]

# Create heatmap data
heatmap_data = avg_probs[unique_indices, :]

# Plot heatmap
im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis', interpolation='nearest')

# Set labels
token_labels = [vocabulary[idx] for idx in unique_indices]
ax.set_yticks(range(len(unique_indices)))
ax.set_yticklabels(token_labels, fontsize=8)
ax.set_xticks(range(n_steps))
ax.set_xticklabels(range(n_steps))

ax.set_xlabel('Time Step', fontsize=12)
ax.set_ylabel('Token', fontsize=12)
ax.set_title(f'Token Probability Heatmap\nPrompt: "{prompt}"', fontsize=14)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Probability', fontsize=12)

# Highlight the best path
for t, token_id in enumerate(best_tokens):
    if token_id in unique_indices:
        y_pos = unique_indices.index(token_id)
        ax.plot(t, y_pos, 'r*', markersize=10)

plt.tight_layout()
heatmap_path = os.path.join(output_dir, "heatmap.png")
fig.savefig(heatmap_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved heatmap to {heatmap_path}")

print("\nVisualization complete!")
print(f"Files saved to: {output_dir}")