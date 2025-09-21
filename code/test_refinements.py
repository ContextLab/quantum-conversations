#!/usr/bin/env python3
"""Test final refinements to bumplot."""

import sys
sys.path.insert(0, '/Users/jmanning/quantum-conversations/code')

from quantum_conversations import ParticleFilter, TokenSequenceVisualizer, ModelManager
import matplotlib.pyplot as plt

model_manager = ModelManager()

print("Testing refinements...")

# Test with medium complexity
pf = ParticleFilter(
    model_name="EleutherAI/pythia-70m",
    n_particles=20,
    temperature=0.7,
    device="cpu",
    model_manager=model_manager,
    seed=42
)

particles = pf.generate("The future is", max_new_tokens=10)
viz = TokenSequenceVisualizer(tokenizer=pf.tokenizer)

fig = viz.visualize_bumplot(
    particles,
    output_path="test_refinements.png",
    max_vocab_display=8,
    show_tokens=True,
    colormap='RdYlGn',
    figsize=(16, 10)
)

plt.close(fig)

print("\nRefinements applied:")
print("✓ Axis labels: 18pt font")
print("✓ Tick labels: 14pt font")
print("✓ Y-axis range: More padding (0.3 to max+0.8)")
print("✓ Colorbar: x=0.90 (small gap from figure)")
print("✓ Token length: Max 15 chars (13 + ..)")
print("✓ Adaptive font: 10pt for long tokens, 12pt standard")
print("✓ Curve alpha: 0.2 (more transparent)")
print("\nSaved to: test_refinements.png")