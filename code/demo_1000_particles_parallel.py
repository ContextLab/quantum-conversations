"""
Demo with 1000 particles running 20 at a time in parallel.
Saves full V×t×n tensor of token probabilities.
"""

import os
import sys
sys.path.append('.')

import numpy as np
import pickle
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from quantum_conversations import ParticleFilter, TokenSequenceVisualizer

# Global lock for file operations
file_lock = Lock()

def generate_single_particle(particle_id, model_name, prompt, max_new_tokens, temperature, top_k, top_p, device, output_dir):
    """Generate a single particle and save its probability distribution."""
    try:
        # Create particle filter for this thread
        pf = ParticleFilter(
            model_name=model_name,
            n_particles=1,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device
        )
        
        # Generate particle
        particles = pf.generate(prompt, max_new_tokens=max_new_tokens)
        particle = particles[0]
        
        # Get vocabulary size
        vocab_size = pf.tokenizer.vocab_size
        
        # Create V×t matrix for this particle
        time_steps = len(particle.token_probs_history)
        prob_matrix = np.zeros((vocab_size, time_steps), dtype=np.float32)
        
        # Fill probability matrix
        for t, step_probs in enumerate(particle.token_probs_history):
            for token_id, prob in step_probs.items():
                if token_id < vocab_size:
                    prob_matrix[token_id, t] = prob
        
        # Save particle data
        particle_data = {
            'particle_id': particle_id,
            'prob_matrix': prob_matrix,
            'tokens': particle.tokens,
            'log_prob': particle.log_prob
        }
        
        # Save to temporary file
        temp_file = os.path.join(output_dir, f'particle_{particle_id:04d}.pkl')
        with file_lock:
            with open(temp_file, 'wb') as f:
                pickle.dump(particle_data, f)
        
        return particle_id, True, particle.tokens, particle.log_prob
        
    except Exception as e:
        print(f"Error in particle {particle_id}: {e}")
        return particle_id, False, None, None

print("=== Quantum Conversations Demo: 1000 Particles in Parallel ===\n")

# Parameters
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
n_particles = 1000
n_parallel = 20  # Run 20 particles at a time
temperature = 1.0
top_k = 100
top_p = 0.95
device = "cpu"
max_new_tokens = 20

# Output directory
output_dir = "../data/derivatives/particle_visualizations/demo_1000_parallel"
os.makedirs(output_dir, exist_ok=True)
temp_dir = os.path.join(output_dir, "temp")
os.makedirs(temp_dir, exist_ok=True)

# Get tokenizer for vocabulary
print("Loading tokenizer...")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
vocab_size = tokenizer.vocab_size

# Save vocabulary as list of strings
print(f"Saving vocabulary ({vocab_size} tokens)...")
vocabulary = []
for token_id in range(vocab_size):
    try:
        token_str = tokenizer.decode([token_id])
        vocabulary.append(token_str)
    except:
        vocabulary.append(f"<UNK_{token_id}>")

# Test prompt
prompt = "The most surprising thing was "
print(f"\nGenerating 1000 particles for: '{prompt}'")
print(f"Running {n_parallel} particles in parallel...")
print("Each particle will generate 20 tokens...")
print("This will take several minutes...\n")

# Run particles in parallel
all_tokens = []
all_log_probs = []
completed = 0

with ThreadPoolExecutor(max_workers=n_parallel) as executor:
    # Submit all tasks
    futures = []
    for i in range(n_particles):
        future = executor.submit(
            generate_single_particle,
            i, model_name, prompt, max_new_tokens,
            temperature, top_k, top_p, device, temp_dir
        )
        futures.append(future)
    
    # Process completed tasks
    for future in as_completed(futures):
        particle_id, success, tokens, log_prob = future.result()
        if success:
            all_tokens.append(tokens)
            all_log_probs.append(log_prob)
            completed += 1
            if completed % 50 == 0:
                print(f"Completed {completed}/{n_particles} particles...")

print(f"\nCompleted all {completed} particles!")

# Load all particle files and create V×t×n tensor
print("\nAggregating into V×t×n tensor...")

# Determine time steps from first particle
with open(os.path.join(temp_dir, 'particle_0000.pkl'), 'rb') as f:
    first_particle = pickle.load(f)
    time_steps = first_particle['prob_matrix'].shape[1]

# Create V×t×n tensor
print(f"Creating tensor of shape ({vocab_size}, {time_steps}, {n_particles})...")
full_tensor = np.zeros((vocab_size, time_steps, n_particles), dtype=np.float32)

# Load each particle and add to tensor
particles_data = []
for i in range(n_particles):
    temp_file = os.path.join(temp_dir, f'particle_{i:04d}.pkl')
    if os.path.exists(temp_file):
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
    'n_particles': len(particles_data),
    'particles_tokens': [p['tokens'] for p in particles_data],  # Token sequences
    'particles_log_probs': [p['log_prob'] for p in particles_data]  # Log probabilities
}

output_file = os.path.join(output_dir, 'token_probabilities_tensor.pkl')
with open(output_file, 'wb') as f:
    pickle.dump(aggregated_data, f)

print(f"✓ Saved aggregated tensor to: {output_file}")
print(f"  Tensor shape: {full_tensor.shape}")
print(f"  File size: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")

# Create visualization using the particles data
print("\nCreating visualizations...")

# Convert back to particle objects for visualization
from quantum_conversations.particle_filter import Particle
particles = []
for i, pdata in enumerate(particles_data):
    # Reconstruct token_probs_history from tensor
    token_probs_history = []
    for t in range(time_steps):
        step_probs = {}
        for v in range(vocab_size):
            if full_tensor[v, t, i] > 0:
                step_probs[v] = float(full_tensor[v, t, i])
        token_probs_history.append(step_probs)
    
    particle = Particle(
        tokens=pdata['tokens'],
        log_prob=pdata['log_prob'],
        token_probs_history=token_probs_history
    )
    particles.append(particle)

# Create visualizer
viz = TokenSequenceVisualizer(
    tokenizer=tokenizer,
    figsize=(20, 12),
    alpha=0.01,
    line_width=0.1
)

# Sankey diagram
fig = viz.visualize(
    particles=particles,
    prompt=prompt,
    output_path=os.path.join(output_dir, "sankey_1000_particles.png"),
    title="Token Generation Paths: 1000 Particles (Parallel Generation)"
)
plt.close(fig)
print("✓ Saved sankey_1000_particles.png")

# Heatmap
fig = viz.visualize_probability_heatmap(
    particles=particles,
    prompt=prompt,
    output_path=os.path.join(output_dir, "heatmap_1000_particles.png"),
    vocab_size=vocab_size
)
plt.close(fig)
print("✓ Saved heatmap_1000_particles.png")

# Analysis
unique_seqs = len(set(tuple(tokens) for tokens in aggregated_data['particles_tokens']))
best_idx = np.argmax(aggregated_data['particles_log_probs'])
best_text = tokenizer.decode(aggregated_data['particles_tokens'][best_idx])

print(f"\nAnalysis:")
print(f"  Most probable sequence: {best_text}")
print(f"  Unique sequences: {unique_seqs}/1000 ({unique_seqs/10:.1f}%)")
print(f"\nAll outputs saved to: {os.path.abspath(output_dir)}")