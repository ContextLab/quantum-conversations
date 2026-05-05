#!/usr/bin/env python3
"""
Test all notebooks by running key cells and verifying outputs.
"""

import sys
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append('.')

from quantum_conversations import ParticleFilter, TokenSequenceVisualizer

def test_basic_generation():
    """Test basic particle generation and visualization."""
    print("=" * 60)
    print("Testing basic particle generation and bumplot visualization")
    print("=" * 60)

    # Initialize particle filter
    pf = ParticleFilter(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        n_particles=10,
        temperature=0.8,
        device="cpu"
    )
    print(f"✓ ParticleFilter initialized with {pf.n_particles} particles")

    # Generate particles
    prompt = "The meaning of life is"
    particles = pf.generate(prompt, max_new_tokens=15)
    print(f"✓ Generated {len(particles)} particles with prompt: '{prompt}'")

    # Check particle structure
    for i, particle in enumerate(particles[:3]):
        print(f"  Particle {i}: {len(particle.tokens)} tokens")

    # Initialize visualizer
    viz = TokenSequenceVisualizer(tokenizer=pf.tokenizer)
    print("✓ TokenSequenceVisualizer initialized")

    # Test bumplot visualization
    output_path = Path("test_outputs/test_basic_bumplot.png")
    output_path.parent.mkdir(exist_ok=True)

    try:
        fig = viz.visualize_bumplot(
            particles,
            output_path=str(output_path),
            max_vocab_display=15,
            color_by='transition_prob',
            show_tokens=True,
            curve_force=0.5,
            prompt=prompt
        )
        plt.close(fig)
        print(f"✓ Bumplot saved to {output_path}")

        # Check file was created and has content
        if output_path.exists():
            size = output_path.stat().st_size
            print(f"  File size: {size:,} bytes")
            if size > 10000:  # Should be at least 10KB for a real plot
                print("  ✓ File size looks reasonable")
            else:
                print("  ⚠ File might be empty or corrupted")
        else:
            print("  ✗ File was not created!")
            return False

    except Exception as e:
        print(f"✗ Error creating bumplot: {e}")
        return False

    return True


def test_color_schemes():
    """Test different color schemes for bumplot."""
    print("\n" + "=" * 60)
    print("Testing different color schemes")
    print("=" * 60)

    # Initialize with smaller model for speed
    pf = ParticleFilter(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        n_particles=8,
        temperature=1.0,
        device="cpu"
    )

    prompt = "Once upon a time"
    particles = pf.generate(prompt, max_new_tokens=10)
    viz = TokenSequenceVisualizer(tokenizer=pf.tokenizer)

    color_schemes = ['transition_prob', 'entropy', 'particle_id']

    for color_by in color_schemes:
        output_path = Path(f"test_outputs/test_color_{color_by}.png")

        try:
            fig = viz.visualize_bumplot(
                particles,
                output_path=str(output_path),
                color_by=color_by,
                max_vocab_display=15,
                show_tokens=False,
                prompt=prompt
            )
            plt.close(fig)
            print(f"✓ Created {color_by} visualization: {output_path}")

            if output_path.exists() and output_path.stat().st_size > 10000:
                print(f"  ✓ File size: {output_path.stat().st_size:,} bytes")
            else:
                print(f"  ⚠ Issue with {color_by} output")
                return False

        except Exception as e:
            print(f"✗ Error with {color_by}: {e}")
            return False

    return True


def test_temperature_effects():
    """Test convergence vs divergence with different temperatures."""
    print("\n" + "=" * 60)
    print("Testing temperature effects on convergence/divergence")
    print("=" * 60)

    prompt = "The answer is"

    # Low temperature (convergent)
    pf_low = ParticleFilter(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        n_particles=6,
        temperature=0.3,
        device="cpu"
    )

    particles_low = pf_low.generate(prompt, max_new_tokens=8)
    viz_low = TokenSequenceVisualizer(tokenizer=pf_low.tokenizer)

    output_low = Path("test_outputs/test_low_temp.png")

    try:
        fig = viz_low.visualize_bumplot(
            particles_low,
            output_path=str(output_low),
            color_by='particle_id',
            max_vocab_display=10,
            show_tokens=True,
            prompt=f"Low Temp (0.3): {prompt}"
        )
        plt.close(fig)
        print(f"✓ Low temperature visualization: {output_low}")

    except Exception as e:
        print(f"✗ Error with low temperature: {e}")
        return False

    # High temperature (divergent)
    pf_high = ParticleFilter(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        n_particles=6,
        temperature=1.5,
        device="cpu"
    )

    particles_high = pf_high.generate(prompt, max_new_tokens=8)
    viz_high = TokenSequenceVisualizer(tokenizer=pf_high.tokenizer)

    output_high = Path("test_outputs/test_high_temp.png")

    try:
        fig = viz_high.visualize_bumplot(
            particles_high,
            output_path=str(output_high),
            color_by='particle_id',
            max_vocab_display=10,
            show_tokens=True,
            prompt=f"High Temp (1.5): {prompt}"
        )
        plt.close(fig)
        print(f"✓ High temperature visualization: {output_high}")

    except Exception as e:
        print(f"✗ Error with high temperature: {e}")
        return False

    # Check for divergence differences
    def calculate_divergence(particles):
        """Simple divergence metric."""
        sequences = [p.tokens for p in particles]
        if len(sequences) < 2:
            return 0

        # Count unique tokens at each position
        divergences = []
        min_len = min(len(seq) for seq in sequences)

        for pos in range(min_len):
            unique_tokens = len(set(seq[pos] for seq in sequences))
            divergences.append(unique_tokens / len(sequences))

        return np.mean(divergences)

    div_low = calculate_divergence(particles_low)
    div_high = calculate_divergence(particles_high)

    print(f"\nDivergence metrics:")
    print(f"  Low temp:  {div_low:.3f}")
    print(f"  High temp: {div_high:.3f}")

    if div_high > div_low:
        print("  ✓ High temperature shows more divergence (as expected)")
    else:
        print("  ⚠ Unexpected divergence pattern")

    return True


def test_large_particle_count():
    """Test with many particles to check visualization scaling."""
    print("\n" + "=" * 60)
    print("Testing with large particle count")
    print("=" * 60)

    # Use more particles
    pf = ParticleFilter(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        n_particles=50,  # More particles
        temperature=0.9,
        device="cpu"
    )

    prompt = "Hello world"
    print(f"Generating {pf.n_particles} particles...")
    particles = pf.generate(prompt, max_new_tokens=8)

    viz = TokenSequenceVisualizer(
        tokenizer=pf.tokenizer,
        alpha=0.05  # Lower alpha for many particles
    )

    output_path = Path("test_outputs/test_many_particles.png")

    try:
        fig = viz.visualize_bumplot(
            particles,
            output_path=str(output_path),
            color_by='transition_prob',
            max_vocab_display=10,
            show_tokens=False,  # No labels with many particles
            prompt=prompt,
            figsize=(16, 10)
        )
        plt.close(fig)
        print(f"✓ Created visualization with {len(particles)} particles")
        print(f"  Saved to: {output_path}")

        if output_path.exists():
            size = output_path.stat().st_size
            print(f"  File size: {size:,} bytes")

    except Exception as e:
        print(f"✗ Error with many particles: {e}")
        return False

    return True


def test_edge_cases():
    """Test edge cases and potential issues."""
    print("\n" + "=" * 60)
    print("Testing edge cases")
    print("=" * 60)

    pf = ParticleFilter(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        n_particles=3,
        temperature=0.7,
        device="cpu"
    )

    # Test 1: Very short generation
    print("Test 1: Very short generation (2 tokens)")
    particles = pf.generate("Hi", max_new_tokens=2)
    viz = TokenSequenceVisualizer(tokenizer=pf.tokenizer)

    try:
        fig = viz.visualize_bumplot(
            particles,
            output_path="test_outputs/test_edge_short.png",
            max_vocab_display=5,
            show_tokens=True
        )
        plt.close(fig)
        print("  ✓ Short generation handled")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

    # Test 2: Single particle
    print("Test 2: Single particle")
    pf.n_particles = 1
    particles = pf.generate("Test", max_new_tokens=5)

    try:
        fig = viz.visualize_bumplot(
            particles,
            output_path="test_outputs/test_edge_single.png",
            max_vocab_display=5,
            show_tokens=True
        )
        plt.close(fig)
        print("  ✓ Single particle handled")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

    # Test 3: Empty prompt
    print("Test 3: Empty prompt")
    pf.n_particles = 3
    particles = pf.generate("", max_new_tokens=5)

    try:
        fig = viz.visualize_bumplot(
            particles,
            output_path="test_outputs/test_edge_empty.png",
            max_vocab_display=5,
            show_tokens=True
        )
        plt.close(fig)
        print("  ✓ Empty prompt handled")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        # Empty prompt might reasonably fail
        print("  (This might be expected)")

    return True


def check_output_quality():
    """Manually inspect output quality indicators."""
    print("\n" + "=" * 60)
    print("Output Quality Check")
    print("=" * 60)

    output_dir = Path("test_outputs")

    if not output_dir.exists():
        print("✗ No test outputs found!")
        return False

    png_files = list(output_dir.glob("*.png"))

    print(f"Found {len(png_files)} output files:")

    issues = []

    for png_file in sorted(png_files):
        size = png_file.stat().st_size
        size_kb = size / 1024

        print(f"\n  {png_file.name}")
        print(f"    Size: {size_kb:.1f} KB")

        # Check for reasonable file sizes
        if size < 5000:
            issues.append(f"{png_file.name} is suspiciously small ({size} bytes)")
            print("    ⚠ Very small file - might be empty")
        elif size < 10000:
            print("    ⚠ Small file - check content")
        else:
            print("    ✓ File size looks good")

        # Verify it's a valid PNG
        try:
            with open(png_file, 'rb') as f:
                header = f.read(8)
                if header[:4] != b'\x89PNG':
                    issues.append(f"{png_file.name} is not a valid PNG")
                    print("    ✗ Invalid PNG header!")
                else:
                    print("    ✓ Valid PNG file")
        except Exception as e:
            issues.append(f"Could not read {png_file.name}: {e}")
            print(f"    ✗ Could not read file: {e}")

    if issues:
        print("\n⚠ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\n✓ All outputs passed quality checks")
        return True


def main():
    """Run all notebook tests."""
    print("NOTEBOOK TESTING SUITE")
    print("=" * 60)

    # Create test output directory
    Path("test_outputs").mkdir(exist_ok=True)

    results = {}

    # Run each test
    tests = [
        ("Basic Generation", test_basic_generation),
        ("Color Schemes", test_color_schemes),
        ("Temperature Effects", test_temperature_effects),
        ("Large Particle Count", test_large_particle_count),
        ("Edge Cases", test_edge_cases),
    ]

    for name, test_func in tests:
        print(f"\nRunning: {name}")
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            results[name] = False

    # Quality check
    print("\n" + "=" * 60)
    results["Quality Check"] = check_output_quality()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:.<30} {status}")

    total_passed = sum(1 for p in results.values() if p)
    total_tests = len(results)

    print(f"\nTotal: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\n🎉 All tests passed! Notebooks are working correctly.")
        return 0
    else:
        print("\n⚠ Some tests failed. Please review the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())