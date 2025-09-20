#!/usr/bin/env python3
"""
Inspect a particle pkl file to understand the data structure
"""

import pickle
import numpy as np

# Load a sample particle file
particle_file = "../data/derivatives/particle_visualizations/demo_1000_resume/temp/particle_0000.pkl"

print(f"Loading particle file: {particle_file}")
with open(particle_file, 'rb') as f:
    data = pickle.load(f)

print(f"\nData type: {type(data)}")
print(f"Data keys: {data.keys() if isinstance(data, dict) else 'Not a dict'}")

if isinstance(data, dict):
    for key, value in data.items():
        print(f"\n{key}:")
        if isinstance(value, (list, np.ndarray)):
            print(f"  Type: {type(value)}")
            print(f"  Length/Shape: {len(value) if isinstance(value, list) else value.shape}")
            if len(value) > 0:
                print(f"  First element: {value[0]}")
                if key == 'token_probs' and isinstance(value, np.ndarray):
                    print(f"  Min: {value.min()}, Max: {value.max()}, Mean: {value.mean()}")
                    print(f"  Non-zero count: {np.count_nonzero(value)}")
        else:
            print(f"  Type: {type(value)}")
            print(f"  Value: {value}")

# Check if the token_probs is the right shape
if 'token_probs' in data:
    token_probs = data['token_probs']
    print(f"\ntoken_probs analysis:")
    print(f"  Shape: {token_probs.shape}")
    print(f"  Expected shape: (32000, 20) for V×t")
    
    # Check each time step
    for t in range(min(5, token_probs.shape[1])):
        probs_at_t = token_probs[:, t]
        top_5_indices = np.argsort(probs_at_t)[-5:][::-1]
        print(f"\n  Time step {t}:")
        print(f"    Sum of probabilities: {probs_at_t.sum()}")
        print(f"    Top 5 token indices: {top_5_indices}")
        print(f"    Top 5 probabilities: {probs_at_t[top_5_indices]}")