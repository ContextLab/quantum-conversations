#!/usr/bin/env python3
"""
Quick test to verify notebooks work with bumplot visualization.
"""

import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append('.')

from quantum_conversations import ParticleFilter, TokenSequenceVisualizer

print("Quick Notebook Test")
print("=" * 50)

# Test basic functionality
print("1. Initializing ParticleFilter...")
pf = ParticleFilter(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    n_particles=5,
    temperature=0.8,
    device="cpu"
)
print("   ✓ ParticleFilter created")

# Generate particles
print("2. Generating particles...")
prompt = "Hello world"
particles = pf.generate(prompt, max_new_tokens=10)
print(f"   ✓ Generated {len(particles)} particles")

# Test visualizer
print("3. Creating visualizer...")
viz = TokenSequenceVisualizer(tokenizer=pf.tokenizer)
print("   ✓ TokenSequenceVisualizer created")

# Create output directory
output_dir = Path("test_outputs")
output_dir.mkdir(exist_ok=True)

# Test bumplot visualization with different color schemes
print("4. Testing bumplot visualizations...")

color_schemes = ['transition_prob', 'entropy', 'particle_id']

for color_by in color_schemes:
    output_path = output_dir / f"quick_test_{color_by}.png"

    try:
        fig = viz.visualize_bumplot(
            particles,
            output_path=str(output_path),
            color_by=color_by,
            max_vocab_display=10,
            show_tokens=(color_by == 'transition_prob'),  # Only show tokens for one
            prompt=f"Test: {color_by}"
        )
        plt.close(fig)

        if output_path.exists():
            size = output_path.stat().st_size
            print(f"   ✓ {color_by}: {size:,} bytes - {output_path.name}")
        else:
            print(f"   ✗ {color_by}: File not created!")

    except Exception as e:
        print(f"   ✗ {color_by}: Error - {e}")

print("\n5. Checking output quality...")
png_files = list(output_dir.glob("quick_test_*.png"))
print(f"   Found {len(png_files)} test outputs")

for png_file in png_files:
    with open(png_file, 'rb') as f:
        header = f.read(4)
        if header == b'\x89PNG':
            print(f"   ✓ {png_file.name} is a valid PNG")
        else:
            print(f"   ✗ {png_file.name} has invalid header")

print("\n✅ Quick test complete! Notebooks should work correctly.")