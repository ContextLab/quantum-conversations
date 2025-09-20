"""
Demo with 100 particles to demonstrate the tensor saving concept.
Runs faster while still showing the V×t×n tensor approach.
"""

import os
import sys
sys.path.append('.')

import numpy as np
import pickle
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

from quantum_conversations import ParticleFilter, TokenSequenceVisualizer

print("=== Quantum Conversations Demo: 100 Particles with Tensor Saving ===\n")

# Parameters
n_particles = 100
temperature = 1.0
top_k = 100
top_p = 0.95
device = "cpu"
max_new_tokens = 20

# Output directory
output_dir = "../data/derivatives/particle_visualizations/demo_100_tensor"
os.makedirs(output_dir, exist_ok=True)
temp_dir = os.path.join(output_dir, "temp")
os.makedirs(temp_dir, exist_ok=True)

# Initialize particle filter
print("Loading model...")
pf = ParticleFilter(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    n_particles=n_particles,
    temperature=temperature,
    top_k=top_k,
    top_p=top_p,
    device=device
)

vocab_size = pf.tokenizer.vocab_size
print(f"Model loaded! Vocabulary size: {vocab_size}")

# Save vocabulary as list of strings
print("Building vocabulary...")
vocabulary = []
for token_id in tqdm(range(vocab_size), desc="Vocabulary"):
    try:
        token_str = pf.tokenizer.decode([token_id])
        vocabulary.append(token_str)
    except:
        vocabulary.append(f"<UNK_{token_id}>")

# Test prompt
prompt = "The most surprising thing was "
print(f"\nGenerating {n_particles} particles for: '{prompt}'")
print("Each particle will generate 20 tokens...\n")

# Generate all particles at once
particles = pf.generate(prompt, max_new_tokens=max_new_tokens)

# Save individual particle files
print("\nSaving individual particle probability distributions...")
time_steps = len(particles[0].token_probs_history) if particles else 0

for i, particle in enumerate(tqdm(particles, desc="Saving particles")):
    # Create V×t matrix for this particle
    prob_matrix = np.zeros((vocab_size, time_steps), dtype=np.float32)
    
    # Fill probability matrix
    for t, step_probs in enumerate(particle.token_probs_history):
        for token_id, prob in step_probs.items():
            if token_id < vocab_size:
                prob_matrix[token_id, t] = prob
    
    # Save particle data
    particle_data = {
        'particle_id': i,
        'prob_matrix': prob_matrix,
        'tokens': particle.tokens,
        'log_prob': particle.log_prob
    }
    
    # Save to temporary file
    temp_file = os.path.join(temp_dir, f'particle_{i:04d}.pkl')
    with open(temp_file, 'wb') as f:
        pickle.dump(particle_data, f)

# Create V×t×n tensor
print(f"\nCreating tensor of shape ({vocab_size}, {time_steps}, {n_particles})...")
full_tensor = np.zeros((vocab_size, time_steps, n_particles), dtype=np.float32)

# Load each particle and add to tensor
particles_data = []
for i in tqdm(range(n_particles), desc="Aggregating tensor"):
    temp_file = os.path.join(temp_dir, f'particle_{i:04d}.pkl')
    with open(temp_file, 'rb') as f:
        particle_data = pickle.load(f)
        full_tensor[:, :, i] = particle_data['prob_matrix']
        particles_data.append(particle_data)
    # Delete temporary file
    os.remove(temp_file)

# Remove temp directory
os.rmdir(temp_dir)

# Save final aggregated data
print("\nSaving aggregated data...")
aggregated_data = {
    'tensor': full_tensor,  # V×t×n array
    'vocabulary': vocabulary,  # List of token strings
    'prompt': prompt,  # Starting sequence
    'vocab_size': vocab_size,
    'time_steps': time_steps,
    'n_particles': n_particles,
    'particles_tokens': [p['tokens'] for p in particles_data],  # Token sequences
    'particles_log_probs': [p['log_prob'] for p in particles_data]  # Log probabilities
}

output_file = os.path.join(output_dir, 'token_probabilities_tensor.pkl')
with open(output_file, 'wb') as f:
    pickle.dump(aggregated_data, f)

print(f"✓ Saved aggregated tensor to: {output_file}")
print(f"  Tensor shape: {full_tensor.shape}")
print(f"  File size: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")

# Create visualizations
print("\nCreating visualizations...")

# Visualizer
viz = TokenSequenceVisualizer(
    tokenizer=pf.tokenizer,
    figsize=(20, 12),
    alpha=0.02,  # Slightly higher for 100 particles
    line_width=0.3
)

# Sankey diagram
fig = viz.visualize(
    particles=particles,
    prompt=prompt,
    output_path=os.path.join(output_dir, "sankey_100_particles.png"),
    title=f"Token Generation Paths: {n_particles} Particles"
)
plt.close(fig)
print("✓ Saved sankey_100_particles.png")

# Heatmap
fig = viz.visualize_probability_heatmap(
    particles=particles,
    prompt=prompt,
    output_path=os.path.join(output_dir, "heatmap_100_particles.png"),
    vocab_size=vocab_size
)
plt.close(fig)
print("✓ Saved heatmap_100_particles.png")

# Analysis
unique_seqs = len(set(tuple(p.tokens) for p in particles))
sequences = pf.get_token_sequences()
log_probs = [lp for _, lp in sequences]
best_idx = np.argmax(log_probs)
best_text = sequences[best_idx][0]

print(f"\nAnalysis:")
print(f"  Most probable sequence: {best_text}")
print(f"  Unique sequences: {unique_seqs}/{n_particles} ({unique_seqs/n_particles*100:.1f}%)")

# Show how to use the saved tensor
print(f"\n=== How to use the saved tensor ===")
print(f"import pickle")
print(f"import numpy as np")
print(f"")
print(f"# Load the data")
print(f"with open('{os.path.basename(output_file)}', 'rb') as f:")
print(f"    data = pickle.load(f)")
print(f"")
print(f"# Access the tensor and metadata")
print(f"tensor = data['tensor']  # Shape: {full_tensor.shape}")
print(f"vocabulary = data['vocabulary']  # List of {len(vocabulary)} token strings")
print(f"prompt = data['prompt']  # '{prompt}'")
print(f"")
print(f"# Example: Get probability of token 'the' at time step 5 for particle 0")
print(f"token_id = vocabulary.index(' the')  # Find token ID")
print(f"prob = tensor[token_id, 5, 0]  # Get probability")
print(f"")
print(f"# Example: Get top 5 most probable tokens at time step 10 across all particles")
print(f"avg_probs = tensor[:, 10, :].mean(axis=1)  # Average across particles")
print(f"top_tokens = np.argsort(avg_probs)[-5:][::-1]  # Get top 5")
print(f"for tid in top_tokens:")
print(f"    print(f'{{vocabulary[tid]}}: {{avg_probs[tid]:.4f}}')")

print(f"\nAll outputs saved to: {os.path.abspath(output_dir)}")