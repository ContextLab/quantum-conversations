"""
Demo with 1000 particles that can resume from existing pickle files.
If a particle's pkl file already exists, load it instead of recomputing.
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

def generate_or_load_particle_batch(particle_ids, model, tokenizer, prompt, max_new_tokens, temperature, top_k, top_p, device, temp_dir):
    """Generate a batch of particles or load from existing files."""
    results = []
    
    for particle_id in particle_ids:
        temp_file = os.path.join(temp_dir, f'particle_{particle_id:04d}.pkl')
        
        # Check if file already exists
        if os.path.exists(temp_file):
            try:
                # Load existing particle
                with open(temp_file, 'rb') as f:
                    particle_data = pickle.load(f)
                results.append((particle_id, True, particle_data['tokens'], particle_data['log_prob'], "loaded"))
                continue
            except Exception as e:
                print(f"\nError loading particle {particle_id}: {e}, regenerating...")
                # File is corrupted, remove and regenerate
                os.remove(temp_file)
        
        # Generate new particle
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
            with open(temp_file, 'wb') as f:
                pickle.dump(particle_data, f)
            
            results.append((particle_id, True, particle.tokens, particle.log_prob, "generated"))
            
        except Exception as e:
            print(f"\nError generating particle {particle_id}: {e}")
            results.append((particle_id, False, None, None, "error"))
    
    return results

print("=== Quantum Conversations Demo: 1000 Particles (Resumable) ===\n")

# Parameters
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
n_particles = 1000
batch_size = 10  # Process 10 particles per thread
n_threads = 1   # Use 1 thread to avoid model loading issues
temperature = 1.0
top_k = 100
top_p = 0.95
device = "cpu"
max_new_tokens = 20

# Output directory
output_dir = "../data/derivatives/particle_visualizations/demo_1000_resume"
os.makedirs(output_dir, exist_ok=True)
temp_dir = os.path.join(output_dir, "temp")
os.makedirs(temp_dir, exist_ok=True)

# Check existing files
existing_files = [f for f in os.listdir(temp_dir) if f.startswith('particle_') and f.endswith('.pkl')]
existing_count = len(existing_files)
print(f"Found {existing_count} existing particle files")

# Test prompt
prompt = "The most surprising thing was "
print(f"Generating {n_particles} particles for: '{prompt}'")
print("Each particle will generate 20 tokens...")

# Only load model if we need to generate new particles
if existing_count < n_particles:
    print(f"\nNeed to generate {n_particles - existing_count} new particles")
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
else:
    print("All particles already exist, loading tokenizer only...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    model = None

# Save vocabulary if it doesn't exist
vocab_file = os.path.join(output_dir, "vocabulary.pkl")
if not os.path.exists(vocab_file):
    print("Building vocabulary...")
    vocabulary = []
    for token_id in tqdm(range(vocab_size), desc="Building vocabulary"):
        try:
            token_str = tokenizer.decode([token_id])
            vocabulary.append(token_str)
        except:
            vocabulary.append(f"<UNK_{token_id}>")
    
    with open(vocab_file, 'wb') as f:
        pickle.dump(vocabulary, f)
    print(f"✓ Saved vocabulary to {vocab_file}")
else:
    print("Loading existing vocabulary...")
    with open(vocab_file, 'rb') as f:
        vocabulary = pickle.load(f)

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
loaded_count = 0
generated_count = 0

if model is not None:
    print(f"\nRunning {batch_size * n_threads} particles in parallel across {n_threads} threads...")
    
    # Process batches sequentially to avoid model sharing issues
    with tqdm(total=n_particles, desc="Processing particles") as pbar:
        for batch in particle_batches:
            results = generate_or_load_particle_batch(
                batch, model, tokenizer, prompt, max_new_tokens,
                temperature, top_k, top_p, device, temp_dir
            )
            for particle_id, success, tokens, log_prob, status in results:
                if success:
                    all_tokens.append(tokens)
                    all_log_probs.append(log_prob)
                    completed += 1
                    if status == "loaded":
                        loaded_count += 1
                    else:
                        generated_count += 1
                else:
                    failed += 1
                pbar.update(1)
else:
    # All particles exist, just load them
    print("\nLoading all existing particles...")
    with tqdm(total=n_particles, desc="Loading particles") as pbar:
        for i in range(n_particles):
            temp_file = os.path.join(temp_dir, f'particle_{i:04d}.pkl')
            if os.path.exists(temp_file):
                try:
                    with open(temp_file, 'rb') as f:
                        particle_data = pickle.load(f)
                    all_tokens.append(particle_data['tokens'])
                    all_log_probs.append(particle_data['log_prob'])
                    completed += 1
                    loaded_count += 1
                except Exception as e:
                    print(f"\nError loading particle {i}: {e}")
                    failed += 1
            else:
                failed += 1
            pbar.update(1)

print(f"\nCompleted {completed} particles:")
print(f"  Loaded existing: {loaded_count}")
print(f"  Generated new: {generated_count}")
print(f"  Failed: {failed}")

if completed == 0:
    print("No particles available. Exiting.")
    sys.exit(1)

# Check if aggregated tensor already exists
tensor_file = os.path.join(output_dir, 'token_probabilities_tensor.pkl')
if os.path.exists(tensor_file) and generated_count == 0:
    print(f"\nAggregated tensor already exists at {tensor_file}")
    print("Loading existing tensor...")
    with open(tensor_file, 'rb') as f:
        aggregated_data = pickle.load(f)
    full_tensor = aggregated_data['tensor']
    print(f"✓ Loaded tensor with shape: {full_tensor.shape}")
else:
    # Create V×t×n tensor
    print("\nAggregating into V×t×n tensor...")
    
    # Find first particle to determine time steps
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
    
    # Create V×t×n tensor for available particles
    print(f"Creating tensor of shape ({vocab_size}, {time_steps}, {completed})...")
    full_tensor = np.zeros((vocab_size, time_steps, completed), dtype=np.float32)
    
    # Load each particle and add to tensor
    particles_data = []
    particle_idx = 0
    
    with tqdm(total=n_particles, desc="Loading particle data") as pbar:
        for i in range(n_particles):
            temp_file = os.path.join(temp_dir, f'particle_{i:04d}.pkl')
            if os.path.exists(temp_file):
                try:
                    with open(temp_file, 'rb') as f:
                        particle_data = pickle.load(f)
                        full_tensor[:, :, particle_idx] = particle_data['prob_matrix']
                        particles_data.append(particle_data)
                        particle_idx += 1
                except Exception as e:
                    print(f"\nError loading particle {i}: {e}")
            pbar.update(1)
    
    # Save final aggregated data
    print("\nSaving aggregated tensor...")
    aggregated_data = {
        'tensor': full_tensor,
        'vocabulary': vocabulary,
        'prompt': prompt,
        'vocab_size': vocab_size,
        'time_steps': time_steps,
        'n_particles': len(particles_data),
        'particles_tokens': [p['tokens'] for p in particles_data],
        'particles_log_probs': [p['log_prob'] for p in particles_data]
    }
    
    with open(tensor_file, 'wb') as f:
        pickle.dump(aggregated_data, f)
    
    print(f"✓ Saved aggregated tensor to: {tensor_file}")

print(f"  Tensor shape: {full_tensor.shape}")
print(f"  File size: {os.path.getsize(tensor_file) / 1024 / 1024:.1f} MB")

# Analysis
unique_seqs = len(set(tuple(tokens) for tokens in aggregated_data['particles_tokens']))
best_idx = np.argmax(aggregated_data['particles_log_probs'])
best_text = tokenizer.decode(aggregated_data['particles_tokens'][best_idx])

print(f"\nAnalysis:")
print(f"  Most probable sequence: {best_text}")
print(f"  Unique sequences: {unique_seqs}/{len(aggregated_data['particles_tokens'])} ({unique_seqs/len(aggregated_data['particles_tokens'])*100:.1f}%)")

print(f"\nAll outputs saved to: {os.path.abspath(output_dir)}")
print(f"To resume: Run this script again and it will use existing particle files")
print(f"To start fresh: Delete the temp/ directory")