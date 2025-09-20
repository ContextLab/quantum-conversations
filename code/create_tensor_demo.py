"""
Quick demo showing how to create and save the V×t×n tensor.
Uses 20 particles for fast execution.
"""

import os
import sys
sys.path.append('.')

import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from quantum_conversations import ParticleFilter

print("=== Creating V×t×n Tensor Demo ===\n")

# Quick demo with 20 particles
pf = ParticleFilter(n_particles=20, temperature=1.0, device="cpu")
prompt = "The most surprising thing was "

print(f"Generating 20 particles for: '{prompt}'")
print("Each particle generates 20 tokens...")

# Generate particles
particles = pf.generate(prompt, max_new_tokens=20)

# Get dimensions
vocab_size = pf.tokenizer.vocab_size
time_steps = len(particles[0].token_probs_history)
n_particles = len(particles)

print(f"\nTensor dimensions:")
print(f"  V (vocabulary): {vocab_size}")
print(f"  t (time steps): {time_steps}")
print(f"  n (particles): {n_particles}")

# Create V×t×n tensor
tensor = np.zeros((vocab_size, time_steps, n_particles), dtype=np.float32)

# Fill tensor with probability data
for p_idx, particle in enumerate(particles):
    for t, step_probs in enumerate(particle.token_probs_history):
        for token_id, prob in step_probs.items():
            if token_id < vocab_size:
                tensor[token_id, t, p_idx] = prob

# Create vocabulary list
vocabulary = []
for token_id in range(vocab_size):
    try:
        vocabulary.append(pf.tokenizer.decode([token_id]))
    except:
        vocabulary.append(f"<UNK_{token_id}>")

# Save data
output_dir = "../data/derivatives/particle_visualizations/tensor_demo"
os.makedirs(output_dir, exist_ok=True)

data = {
    'tensor': tensor,
    'vocabulary': vocabulary,
    'prompt': prompt,
    'vocab_size': vocab_size,
    'time_steps': time_steps,
    'n_particles': n_particles,
    'particles_tokens': [p.tokens for p in particles],
    'particles_log_probs': [p.log_prob for p in particles]
}

output_file = os.path.join(output_dir, 'tensor_demo.pkl')
with open(output_file, 'wb') as f:
    pickle.dump(data, f)

print(f"\n✓ Saved tensor to: {output_file}")
print(f"  File size: {os.path.getsize(output_file) / 1024:.1f} KB")

# Show usage example
print(f"\n=== Usage Example ===")
print("# Load the tensor")
print("import pickle")
print("with open('tensor_demo.pkl', 'rb') as f:")
print("    data = pickle.load(f)")
print("")
print("tensor = data['tensor']")
print("vocabulary = data['vocabulary']")
print("prompt = data['prompt']")
print("")
print("# Example: Get probability of token 'that' at time step 3 for particle 0")
print("try:")
print("    token_id = vocabulary.index(' that')")
print("    prob = tensor[token_id, 3, 0]")
print("    print(f'P(that | t=3, particle=0) = {prob:.4f}')")
print("except ValueError:")
print("    print('Token not found')")

# Actually run the example
try:
    token_id = vocabulary.index(' that')
    prob = tensor[token_id, 3, 0]
    print(f"\nActual result: P(' that' | t=3, particle=0) = {prob:.4f}")
except ValueError:
    print("\nToken ' that' not found in vocabulary")

print(f"\nTensor shape: {tensor.shape}")
print(f"Non-zero entries: {np.count_nonzero(tensor)} / {tensor.size}")
print(f"Sparsity: {(1 - np.count_nonzero(tensor)/tensor.size)*100:.1f}%")

print("\n✓ Demo complete!")