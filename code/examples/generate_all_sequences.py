#!/usr/bin/env python3
"""
Generate particle data and visualizations for all 20-30 input sequences
"""

import os
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from quantum_conversations.particle_filter import ParticleFilter
from quantum_conversations.visualizer import TokenSequenceVisualizer
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define prompts with varying ambiguity levels
prompts = {
    "high_ambiguity": [
        "The most surprising thing was ",
        "It all started when ",
        "I couldn't believe that ",
        "The decision was ",
        "Something unexpected happened: ",
        "Without warning, the ",
        "In that moment, everything ",
        "The truth about ",
    ],
    "medium_ambiguity": [
        "The scientific experiment revealed ",
        "The recipe called for ",
        "The teacher explained that ",
        "The weather forecast predicted ",
        "The company announced ",
        "The research showed ",
        "The artist created ",
        "The athlete trained ",
    ],
    "low_ambiguity": [
        "Two plus two equals ",
        "The capital of France is ",
        "Water freezes at ",
        "The sun rises in the ",
        "A triangle has ",
        "The alphabet starts with ",
        "January comes after ",
        "Red mixed with blue makes ",
    ]
}

# Parameters
n_particles = 100  # Reduced for faster generation
max_new_tokens = 20
temperature = 1.0
top_k = 100
top_p = 0.95
device = "cpu"

# Output directory
base_output_dir = "../data/derivatives/particle_visualizations/all_sequences"
os.makedirs(base_output_dir, exist_ok=True)

# Load model and tokenizer once
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"Loading model and tokenizer from {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map=device
)
model.eval()

# Create visualizer
visualizer = TokenSequenceVisualizer(
    tokenizer=tokenizer,
    figsize=(20, 12),
    max_tokens_display=50,
    alpha=0.01,
    line_width=0.5
)

# Process each ambiguity level
for ambiguity_level, prompt_list in prompts.items():
    print(f"\n{'='*60}")
    print(f"Processing {ambiguity_level} prompts")
    print(f"{'='*60}")
    
    level_dir = os.path.join(base_output_dir, ambiguity_level)
    os.makedirs(level_dir, exist_ok=True)
    
    # Process each prompt
    for prompt_idx, prompt in enumerate(prompt_list):
        print(f"\n[{prompt_idx+1}/{len(prompt_list)}] Processing: '{prompt}'")
        
        # Create prompt-specific directory
        safe_prompt = prompt.strip().replace(' ', '_')[:30]
        prompt_dir = os.path.join(level_dir, f"{prompt_idx:02d}_{safe_prompt}")
        os.makedirs(prompt_dir, exist_ok=True)
        
        # Check if already processed
        tensor_path = os.path.join(prompt_dir, "tensor.pkl")
        if os.path.exists(tensor_path):
            print(f"  Already processed, loading existing data...")
            with open(tensor_path, 'rb') as f:
                tensor_data = pickle.load(f)
            particles = tensor_data['particles']
        else:
            # Create particle filter
            print(f"  Generating {n_particles} particles...")
            pf = ParticleFilter(
                model_name=model_name,
                n_particles=n_particles,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                device=device
            )
            
            # Use pre-loaded model and tokenizer
            pf.model = model
            pf.tokenizer = tokenizer
            
            # Generate particles
            particles = pf.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                show_progress=True
            )
            
            # Save tensor data
            print(f"  Saving tensor data...")
            # Create V×t×n tensor
            vocab_size = len(tokenizer)
            tensor = np.zeros((vocab_size, max_new_tokens, n_particles), dtype=np.float32)
            
            for p_idx, particle in enumerate(particles):
                # Get token probability history
                for t_idx, token_probs in enumerate(particle.token_probs_history[:max_new_tokens]):
                    for token_id, prob in token_probs.items():
                        if token_id < vocab_size:
                            tensor[token_id, t_idx, p_idx] = prob
            
            # Save data
            tensor_data = {
                'tensor': tensor,
                'particles': particles,
                'prompt': prompt,
                'n_particles': n_particles,
                'max_new_tokens': max_new_tokens,
                'ambiguity_level': ambiguity_level
            }
            
            with open(tensor_path, 'wb') as f:
                pickle.dump(tensor_data, f)
        
        # Create visualizations
        print(f"  Creating visualizations...")
        
        # Sankey diagram
        fig_sankey = visualizer.visualize(
            particles=particles,
            prompt=prompt,
            title=f"{ambiguity_level.replace('_', ' ').title()}: '{prompt}'",
            highlight_most_probable=True
        )
        sankey_path = os.path.join(prompt_dir, "sankey.png")
        fig_sankey.savefig(sankey_path, dpi=300, bbox_inches='tight')
        plt.close(fig_sankey)
        
        # Heatmap
        # Average probabilities across particles
        avg_probs = np.mean(tensor_data['tensor'], axis=2)
        
        # Create custom heatmap
        fig_heat, ax = plt.subplots(figsize=(16, 10))
        
        # Select top tokens to display
        n_display = 50
        top_tokens = set()
        for t in range(max_new_tokens):
            top_indices = np.argsort(avg_probs[:, t])[-10:][::-1]
            top_tokens.update(top_indices)
        
        top_tokens = sorted(list(top_tokens))[:n_display]
        heatmap_data = avg_probs[top_tokens, :]
        
        # Use log scale for better visibility
        heatmap_data_log = np.log10(heatmap_data + 1e-10)
        im = ax.imshow(heatmap_data_log, aspect='auto', cmap='viridis')
        
        # Labels
        vocab = [tokenizer.decode([i]) for i in range(len(tokenizer))]
        token_labels = [vocab[idx] if idx < len(vocab) else f"[{idx}]" for idx in top_tokens]
        token_labels = [t[:15] + "..." if len(t) > 15 else t for t in token_labels]
        
        ax.set_yticks(range(len(top_tokens)))
        ax.set_yticklabels(token_labels, fontsize=6)
        ax.set_xticks(range(max_new_tokens))
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Token', fontsize=12)
        ax.set_title(f'Token Probability Heatmap\n{ambiguity_level.replace("_", " ").title()}: "{prompt}"', fontsize=14)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('log10(Probability)', fontsize=12)
        
        plt.tight_layout()
        heatmap_path = os.path.join(prompt_dir, "heatmap.png")
        fig_heat.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close(fig_heat)
        
        print(f"  Saved visualizations to {prompt_dir}")

print("\n" + "="*60)
print("All sequences processed!")
print(f"Results saved to: {base_output_dir}")
print("="*60)