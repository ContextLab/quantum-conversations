# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Quantum Conversations explores whether the "paths not taken" in language generation (tokens that could have been output but weren't) affect future outputs. The project uses particle filters to approximate the full set of possible generation paths, storing V×t×n tensors (vocabulary size × time steps × particles) to track token probability histories.

## Commands

### Environment Setup
```bash
# Initialize git submodules (required for bibliography)
./setup.sh

# Build Docker environment (for paper/notebooks)
docker build -t quantum-conversations .

# Install Python package locally
cd code
pip install -r requirements.txt
pip install -e .
```

### Running Tests
```bash
cd code
pytest tests/ -v
# Run with coverage
pytest tests/ --cov=quantum_conversations
```

### Paper Compilation
```bash
cd paper
./compile.sh
# Generates main.pdf and supplement.pdf
```

### Running Demos
```bash
# Various demo scripts for particle filter experiments
python code/demo_200_particles.py
python code/demo_1000_particles_final.py
python code/run_1000_particles_demo.py
```

## Architecture

### Core Components

**quantum_conversations.ParticleFilter**
- Tracks multiple generation hypotheses simultaneously using particle filtering
- Records full token probability distributions at each timestep
- Supports temperature, top-k, and top-p sampling strategies
- Handles model loading from HuggingFace (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)

**quantum_conversations.TokenSequenceVisualizer**
- Creates Sankey-like diagrams showing particle paths through token space
- Generates probability heatmaps for analyzing generation patterns
- Supports batch visualization of multiple prompts
- Saves outputs to `/data/derivatives/` directory

### Key Data Structures
- Particles store: token IDs, log probabilities, full probability distributions
- Tensors are V×t×n dimensional (vocabulary × timesteps × particles)
- Visualization data saved as .pkl files for reproducibility

### Directory Layout
- `/code/quantum_conversations/` - Main Python package
- `/code/tests/` - pytest test suite
- `/code/notebooks/` - Jupyter notebooks (quantum_conversations_demo.ipynb)
- `/data/raw/` - Original experimental data
- `/data/derivatives/` - Processed results and visualizations
- `/paper/` - LaTeX source with CDL-bibliography submodule