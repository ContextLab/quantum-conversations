"""
Demo with 1000 particles running 20 at a time in parallel.
Saves full V×t×n tensor of token probabilities.
Fixed version with shared model and tqdm progress.
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
from tqdm import tqdm

from quantum_conversations import ParticleFilter, TokenSequenceVisualizer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Global lock for thread safety
generation_lock = Lock()

def generate_single_particle_batch(particle_ids, model, tokenizer, prompt, max_new_tokens, temperature, top_k, top_p, device, output_dir):
    """Generate a batch of particles and save their probability distributions."""
    results = []
    
    for particle_id in particle_ids:
        try:
            # Create particle filter using shared model
            pf = ParticleFilter(
                model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                n_particles=1,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                device=device
            )
            # Replace model and tokenizer with shared ones
            pf.model = model
            pf.tokenizer = tokenizer
            
            # Generate particle
            with generation_lock:
                particles = pf.generate(prompt, max_new_tokens=max_new_tokens)
            
            particle = particles[0]
            
            # Get vocabulary size
            vocab_size = tokenizer.vocab_size
            
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
            with open(temp_file, 'wb') as f:
                pickle.dump(particle_data, f)
            
            results.append((particle_id, True, particle.tokens, particle.log_prob))
            
        except Exception as e:
            print(f"\nError in particle {particle_id}: {e}")
            results.append((particle_id, False, None, None))
    
    return results

print("=== Quantum Conversations Demo: 1000 Particles in Parallel ===\n")

# Parameters
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
n_particles = 1000
batch_size = 5  # Process 5 particles per thread
n_threads = 4   # Use 4 threads (20 particles total in parallel)
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

# Load model and tokenizer once
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
).to(device)
model.eval()

vocab_size = tokenizer.vocab_size
print(f"Model loaded! Vocabulary size: {vocab_size}")

# Save vocabulary as list of strings
print("Saving vocabulary...")
vocabulary = []
for token_id in tqdm(range(vocab_size), desc="Building vocabulary"):
    try:
        token_str = tokenizer.decode([token_id])
        vocabulary.append(token_str)
    except:
        vocabulary.append(f"<UNK_{token_id}>")

# Test prompt
prompt = "The most surprising thing was "
print(f"\nGenerating {n_particles} particles for: '{prompt}'")
print(f"Running {batch_size * n_threads} particles in parallel across {n_threads} threads...")
print("Each particle will generate 20 tokens...\n")

# Create batches
particle_batches = []
for i in range(0, n_particles, batch_size):
    batch = list(range(i, min(i + batch_size, n_particles)))
    particle_batches.append(batch)

# Run particles in parallel batches
all_tokens = []
all_log_probs = []
completed = 0
failed = 0

with ThreadPoolExecutor(max_workers=n_threads) as executor:
    # Submit batches with progress bar
    futures = []
    with tqdm(total=len(particle_batches), desc="Submitting batches") as pbar:
        for batch in particle_batches:
            future = executor.submit(
                generate_single_particle_batch,
                batch, model, tokenizer, prompt, max_new_tokens,
                temperature, top_k, top_p, device, temp_dir
            )
            futures.append((future, batch))
            pbar.update(1)
    
    # Process completed batches with progress bar
    with tqdm(total=n_particles, desc="Generating particles") as pbar:
        for future, batch in futures:
            results = future.result()
            for particle_id, success, tokens, log_prob in results:
                if success:
                    all_tokens.append(tokens)
                    all_log_probs.append(log_prob)
                    completed += 1
                else:
                    failed += 1
                pbar.update(1)

print(f"\nCompleted {completed} particles successfully, {failed} failed")

if completed == 0:
    print("No particles generated successfully. Exiting.")
    sys.exit(1)

# Load all particle files and create V×t×n tensor
print("\nAggregating into V×t×n tensor...")

# Find first successful particle to determine time steps
first_particle_file = None
for i in range(n_particles):
    temp_file = os.path.join(temp_dir, f'particle_{i:04d}.pkl')
    if os.path.exists(temp_file):
        first_particle_file = temp_file
        break

if first_particle_file is None:
    print("No particle files found. Exiting.")
    sys.exit(1)

with open(first_particle_file, 'rb') as f:
    first_particle = pickle.load(f)
    time_steps = first_particle['prob_matrix'].shape[1]

# Create V×t×n tensor for successful particles
print(f"Creating tensor of shape ({vocab_size}, {time_steps}, {completed})...")
full_tensor = np.zeros((vocab_size, time_steps, completed), dtype=np.float32)

# Load each particle and add to tensor
particles_data = []
particle_idx = 0

with tqdm(total=n_particles, desc="Loading particle data") as pbar:
    for i in range(n_particles):
        temp_file = os.path.join(temp_dir, f'particle_{i:04d}.pkl')
        if os.path.exists(temp_file):
            with open(temp_file, 'rb') as f:
                particle_data = pickle.load(f)
                full_tensor[:, :, particle_idx] = particle_data['prob_matrix']
                particles_data.append(particle_data)
                particle_idx += 1
            # Delete temporary file
            os.remove(temp_file)
        pbar.update(1)

# Remove temp directory
try:
    os.rmdir(temp_dir)
except:
    pass

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

with tqdm(total=len(particles_data), desc="Reconstructing particles") as pbar:
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
        pbar.update(1)

# Create visualizer
viz = TokenSequenceVisualizer(
    tokenizer=tokenizer,
    figsize=(20, 12),
    alpha=0.01,
    line_width=0.1
)

# Sankey diagram
print("Creating Sankey diagram...")
fig = viz.visualize(
    particles=particles,
    prompt=prompt,
    output_path=os.path.join(output_dir, "sankey_1000_particles.png"),
    title=f"Token Generation Paths: {completed} Particles (Parallel Generation)"
)
plt.close(fig)
print("✓ Saved sankey_1000_particles.png")

# Heatmap
print("Creating probability heatmap...")
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
print(f"  Unique sequences: {unique_seqs}/{completed} ({unique_seqs/completed*100:.1f}%)")
print(f"\nAll outputs saved to: {os.path.abspath(output_dir)}")
print(f"\nThe saved tensor can be loaded with:")
print(f"  with open('{output_file}', 'rb') as f:")
print(f"      data = pickle.load(f)")
print(f"  tensor = data['tensor']  # Shape: {full_tensor.shape}")
print(f"  vocabulary = data['vocabulary']  # List of {len(vocabulary)} token strings")
print(f"  prompt = data['prompt']  # '{prompt}'")