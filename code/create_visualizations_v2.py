#!/usr/bin/env python3
"""
Create visualizations from existing pkl files - Version 2
Properly handles the actual data structure
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from quantum_conversations.visualizer import TokenSequenceVisualizer
from quantum_conversations.particle_filter import Particle

# Parameters
data_dir = "../data/derivatives/particle_visualizations/demo_1000_resume"
temp_dir = os.path.join(data_dir, "temp")
output_dir = os.path.join(data_dir, "figures_v2")
os.makedirs(output_dir, exist_ok=True)

# Load vocabulary
vocab_path = os.path.join(data_dir, "vocabulary.pkl")
print("Loading vocabulary...")
with open(vocab_path, 'rb') as f:
    vocabulary = pickle.load(f)

# Get list of particle files
particle_files = [f for f in os.listdir(temp_dir) if f.startswith('particle_') and f.endswith('.pkl')]
particle_files.sort()
print(f"\nFound {len(particle_files)} particle files")

# Load all particles
particles_data = []
for pfile in particle_files[:139]:  # Use only the 139 we know exist
    fpath = os.path.join(temp_dir, pfile)
    with open(fpath, 'rb') as f:
        data = pickle.load(f)
        particles_data.append(data)

print(f"Loaded {len(particles_data)} particles")

# Extract prompt from first particle's tokens
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"\nLoading tokenizer from {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Decode the first 7 tokens as the prompt (based on "The most surprising thing was ")
prompt_tokens = particles_data[0]['tokens'][:7]
prompt = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
print(f"Prompt: '{prompt}'")

# Extract generated tokens (after prompt)
prompt_length = 7
n_generated = 20

# Create arrays for visualization
all_sequences = []
all_probs = []
all_log_probs = []

for pdata in particles_data:
    # Extract generated tokens (skip prompt)
    full_tokens = pdata['tokens']
    generated_tokens = full_tokens[prompt_length:prompt_length + n_generated]
    
    # Pad if necessary
    if len(generated_tokens) < n_generated:
        generated_tokens.extend([0] * (n_generated - len(generated_tokens)))
    
    all_sequences.append(generated_tokens)
    
    # Extract probabilities for the generated tokens
    prob_matrix = pdata['prob_matrix']
    probs = []
    for t in range(n_generated):
        if t < prob_matrix.shape[1] and t < len(generated_tokens):
            token_id = generated_tokens[t]
            prob = prob_matrix[token_id, t]
            probs.append(prob if prob > 0 else 1e-10)  # Avoid log(0)
        else:
            probs.append(1e-10)
    
    all_probs.append(probs)
    all_log_probs.append(pdata['log_prob'])

# Find best particle
best_idx = np.argmax(all_log_probs)
best_sequence = all_sequences[best_idx]
best_text = tokenizer.decode(best_sequence, skip_special_tokens=True)
print(f"\nBest particle: {best_idx}")
print(f"Best continuation: '{best_text}'")
print(f"Full text: '{prompt}{best_text}'")

# Create visualizer
visualizer = TokenSequenceVisualizer(
    tokenizer=tokenizer,
    figsize=(20, 12),
    max_tokens_display=50,
    alpha=0.01,
    line_width=0.5
)

# Create Particle objects
particles = []
for i in range(len(particles_data)):
    particle = Particle(
        tokens=all_sequences[i],
        log_prob=all_log_probs[i],
        token_probs_history=[]
    )
    particles.append(particle)

# Create Sankey diagram
print("\nCreating Sankey diagram...")
fig = visualizer.visualize(
    particles=particles,
    prompt=prompt,
    output_path=None,
    title="Token Generation Paths (139 particles)",
    highlight_most_probable=True
)
sankey_path = os.path.join(output_dir, "sankey_diagram.png")
fig.savefig(sankey_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {sankey_path}")

# Create heatmap showing full probability distributions
print("\nCreating heatmap...")
fig, ax = plt.subplots(figsize=(20, 12))

# Aggregate probability matrices from all particles
aggregated_probs = np.zeros((32000, n_generated))
for i, pdata in enumerate(particles_data):
    prob_matrix = pdata['prob_matrix']
    for t in range(min(n_generated, prob_matrix.shape[1])):
        aggregated_probs[:, t] += prob_matrix[:, t]

# Average the probabilities
aggregated_probs /= len(particles_data)

# Select top tokens to display
n_display = 100
top_tokens_per_step = set()
for t in range(n_generated):
    top_indices = np.argsort(aggregated_probs[:, t])[-10:][::-1]
    top_tokens_per_step.update(top_indices)

# Convert to sorted list and limit
top_tokens = sorted(list(top_tokens_per_step))[:n_display]

# Create heatmap data
heatmap_data = aggregated_probs[top_tokens, :]

# Plot with log scale for better visibility
heatmap_data_log = np.log10(heatmap_data + 1e-10)
im = ax.imshow(heatmap_data_log, aspect='auto', cmap='viridis', interpolation='nearest')

# Labels
token_labels = []
for idx in top_tokens:
    token = vocabulary[idx] if idx < len(vocabulary) else f"[{idx}]"
    # Truncate long tokens
    if len(token) > 15:
        token = token[:12] + "..."
    token_labels.append(token)

ax.set_yticks(range(len(top_tokens)))
ax.set_yticklabels(token_labels, fontsize=6)
ax.set_xticks(range(n_generated))
ax.set_xticklabels(range(n_generated))

ax.set_xlabel('Generated Token Position', fontsize=12)
ax.set_ylabel('Token', fontsize=12)
ax.set_title(f'Token Probability Heatmap (log scale)\nPrompt: "{prompt}"\n139 particles', fontsize=14)

# Colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('log10(Probability)', fontsize=12)

# Highlight best path
for t, token_id in enumerate(best_sequence):
    if token_id in top_tokens:
        y_pos = top_tokens.index(token_id)
        ax.plot(t, y_pos, 'r*', markersize=8)

plt.tight_layout()
heatmap_path = os.path.join(output_dir, "heatmap.png")
fig.savefig(heatmap_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {heatmap_path}")

# Create a focused heatmap showing only the most probable tokens
print("\nCreating focused heatmap...")
fig, ax = plt.subplots(figsize=(16, 10))

# For each time step, show only top 20 tokens
n_top = 20
focused_data = []
focused_labels = []

for t in range(n_generated):
    probs_at_t = aggregated_probs[:, t]
    top_indices = np.argsort(probs_at_t)[-n_top:][::-1]
    
    for idx in top_indices:
        focused_data.append(probs_at_t[idx])
        token = vocabulary[idx] if idx < len(vocabulary) else f"[{idx}]"
        if len(token) > 20:
            token = token[:17] + "..."
        focused_labels.append(f"t{t}: {token}")

# Reshape for visualization
focused_matrix = np.array(focused_data).reshape(n_generated, n_top).T

im = ax.imshow(focused_matrix, aspect='auto', cmap='hot', interpolation='nearest')
ax.set_yticks(range(n_top))
ax.set_yticklabels([f"Rank {i+1}" for i in range(n_top)], fontsize=8)
ax.set_xticks(range(n_generated))
ax.set_xticklabels(range(n_generated))

ax.set_xlabel('Generated Token Position', fontsize=12)
ax.set_ylabel('Probability Rank', fontsize=12)
ax.set_title(f'Top Token Probabilities by Position\nPrompt: "{prompt}"', fontsize=14)

# Add text annotations for top 3 at each position
for t in range(n_generated):
    probs_at_t = aggregated_probs[:, t]
    top_indices = np.argsort(probs_at_t)[-3:][::-1]
    
    for rank, idx in enumerate(top_indices):
        token = vocabulary[idx] if idx < len(vocabulary) else f"[{idx}]"
        if len(token) > 8:
            token = token[:5] + ".."
        prob = probs_at_t[idx]
        ax.text(t, rank, f"{token}\n{prob:.3f}", 
                ha='center', va='center', fontsize=6,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Probability', fontsize=12)

plt.tight_layout()
focused_path = os.path.join(output_dir, "heatmap_focused.png")
fig.savefig(focused_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {focused_path}")

print("\nVisualization complete!")
print(f"All files saved to: {output_dir}")