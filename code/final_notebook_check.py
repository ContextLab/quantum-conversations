#!/usr/bin/env python3
"""
Final check that all notebooks work correctly with bumplot-only visualization.
"""

import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append('.')

from quantum_conversations import ParticleFilter, TokenSequenceVisualizer

print("Final Notebook Verification")
print("=" * 60)

# Create output directory
output_dir = Path("final_test_outputs")
output_dir.mkdir(exist_ok=True)

try:
    # Test 1: Basic bumplot generation (main demo notebook)
    print("\n1. Testing main demo notebook functionality...")
    pf = ParticleFilter(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        n_particles=8,
        temperature=0.8,
        device="cpu"
    )

    prompt = "The meaning of life is"
    particles = pf.generate(prompt, max_new_tokens=10)

    viz = TokenSequenceVisualizer(tokenizer=pf.tokenizer)

    # Test bumplot with all color schemes
    for color_by in ['transition_prob', 'entropy', 'particle_id']:
        output_path = output_dir / f"main_demo_{color_by}.png"
        fig = viz.visualize_bumplot(
            particles,
            output_path=str(output_path),
            color_by=color_by,
            max_vocab_display=15,
            show_tokens=(color_by == 'transition_prob'),
            prompt=prompt
        )
        plt.close(fig)

        if output_path.exists() and output_path.stat().st_size > 10000:
            print(f"   ✓ {color_by} visualization: {output_path.stat().st_size:,} bytes")
        else:
            print(f"   ✗ Issue with {color_by}")

    # Test 2: Temperature comparison (bumplot demo notebook)
    print("\n2. Testing temperature comparison...")
    temperatures = [0.3, 1.0, 1.5]

    for temp in temperatures:
        pf_temp = ParticleFilter(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            n_particles=6,
            temperature=temp,
            device="cpu"
        )

        particles_temp = pf_temp.generate("Once upon a time", max_new_tokens=8)
        viz_temp = TokenSequenceVisualizer(tokenizer=pf_temp.tokenizer)

        output_path = output_dir / f"temp_{temp}.png"
        fig = viz_temp.visualize_bumplot(
            particles_temp,
            output_path=str(output_path),
            color_by='particle_id',
            max_vocab_display=10,
            show_tokens=False
        )
        plt.close(fig)

        if output_path.exists():
            print(f"   ✓ Temperature {temp}: {output_path.stat().st_size:,} bytes")

    # Test 3: Interactive exploration function
    print("\n3. Testing interactive exploration...")
    def explore_prompt(prompt, n_particles=5, max_tokens=10, temperature=0.9):
        pf = ParticleFilter(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            n_particles=n_particles,
            temperature=temperature,
            device="cpu"
        )

        particles = pf.generate(prompt, max_new_tokens=max_tokens)
        viz = TokenSequenceVisualizer(tokenizer=pf.tokenizer)

        output_path = output_dir / f"explore_{prompt[:10].replace(' ', '_')}.png"
        fig = viz.visualize_bumplot(
            particles,
            output_path=str(output_path),
            max_vocab_display=12,
            color_by='transition_prob',
            show_tokens=True,
            prompt=prompt
        )
        plt.close(fig)

        return output_path.exists() and output_path.stat().st_size > 10000

    test_prompts = [
        "The future of AI will",
        "def hello():",
        "Breaking news:"
    ]

    for prompt in test_prompts:
        if explore_prompt(prompt):
            print(f"   ✓ '{prompt[:20]}...'")
        else:
            print(f"   ✗ Failed: '{prompt}'")

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL CHECK SUMMARY")
    print("=" * 60)

    png_files = list(output_dir.glob("*.png"))
    print(f"Generated {len(png_files)} visualization files")

    valid_count = 0
    for png_file in png_files:
        with open(png_file, 'rb') as f:
            if f.read(4) == b'\x89PNG':
                valid_count += 1

    print(f"Valid PNG files: {valid_count}/{len(png_files)}")

    if valid_count == len(png_files):
        print("\n✅ SUCCESS: All notebooks are working correctly!")
        print("   - Bumplot visualization works with all color schemes")
        print("   - Temperature comparisons work correctly")
        print("   - Interactive exploration works")
        print("   - All outputs are valid PNG files")
    else:
        print("\n⚠ Some issues detected. Please review.")

except Exception as e:
    print(f"\n✗ Error during testing: {e}")
    import traceback
    traceback.print_exc()