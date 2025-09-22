# Quantum Conversations

## Overview

Do the things we *don't* say (but perhaps that we *thought*) affect what we say (or think!) in the future?  Modern (standard) LLMs output sequences of tokens, one token at a time. However, in order to emit a single token at timestep $t$, a model carries out a selection by taking a draw over the $V$ possible tokens in the vocabulary. The "chosen" token, $x_t$, will tend to be one of the more probable tokens, but (particularly when the model temperature is high) it might not be the *most* probable token-- and occasionally the chosen token might even be a lower probability token.  Given that we are currently at timepoint $t$, our core question is: do humans "keep around" some representation of the history of "what *could* have been outputted" rather than solely storing the sequence of previously outputted tokens?

## Approach

Given a model, $M$, and a sequence of tokens, $x_1, x_2, ..., x_t$, we want to examine the probability of outputting each possible token (there are $V$ of them) at time $t+1$.  We can then store the full "history" of outputted token probabilities as a $V \times t$ matrix.  In principle, we could consider the full set of branching paths that could have been taken.  However, for a sequence of $t$ tokens, this would require storing $V^t$ possible paths.  This is intractable, even for relatively short sequences ($V$ is on the order of 100,000, and $t$ is on the order of thousands).  Here we approximate the set of possible paths using particle filters.  Then for $n$ particles, we need to store a $V \times t \times n$ tensor.

We can then ask: given an observed sequence of tokens from a human conversation or narrative, can we better explain the token-by-token probabilities using that full tensor (e.g., by accounting for tokens *not* emitted), or is "all" of the predictive power carried solely by the single observed sequence?

## Python Toolbox

This repository includes a Python implementation of the particle filter approach for visualizing token generation paths in language models.

### Installation

```bash
# Clone repository
git clone https://github.com/ContextLab/quantum-conversations.git
cd quantum-conversations

# Install dependencies
cd code
pip install -r requirements.txt
pip install -e .
```

### Project Structure

```
quantum-conversations/
├── code/
│   ├── quantum_conversations/  # Core package
│   │   ├── particle_filter.py  # Particle filtering implementation
│   │   ├── visualizer.py       # Bumplot visualization
│   │   └── custom_bumplot.py   # Advanced bumplot features
│   ├── examples/               # Demo scripts
│   ├── tests/                  # Test suite (real tests, no mocks)
│   ├── scripts/                # Maintenance tools
│   └── notebooks/              # Jupyter notebooks
├── data/
│   ├── raw/                   # Original experimental data
│   └── derivatives/            # Generated visualizations
└── paper/                      # Research paper (LaTeX)
```

### Repository Maintenance

This repository includes automated cleanup and maintenance tools:

```bash
# Run cleanup audit (dry-run mode to see what would change)
python code/scripts/cleanup_repository.py --dry-run

# Execute full cleanup
python code/scripts/cleanup_repository.py

# Install pre-commit hooks for code quality
pre-commit install

# Run all tests with real operations (no mocks)
cd code && pytest tests/ -v
```

### Quick Start

```python
from quantum_conversations import ParticleFilter, TokenSequenceVisualizer

# Initialize particle filter with TinyLlama
pf = ParticleFilter(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    n_particles=10,
    temperature=0.8
)

# Generate particles for a prompt
prompt = "The meaning of life is"
particles = pf.generate(prompt, max_new_tokens=30)

# Visualize the paths
viz = TokenSequenceVisualizer(tokenizer=pf.tokenizer)
fig = viz.visualize(particles, prompt)
```

### Key Components

#### ParticleFilter
- Tracks multiple generation hypotheses simultaneously
- Records token probabilities at each step
- Supports temperature, top-k, and top-p sampling

#### TokenSequenceVisualizer
- Creates Sankey-like diagrams showing particle paths
- **NEW: Bump plot visualizations** showing token trajectories over time
- Generates probability heatmaps
- Customizable styling and output options

### Running Tests

```bash
cd code
pytest tests/ -v
```

### Example Notebook

See `code/quantum_conversations_demo.ipynb` for a comprehensive demonstration including:
- Visualizing paths for 30 different starter sequences
- Analyzing divergence patterns based on prompt ambiguity
- Interactive exploration of custom prompts

### Bump Plot Visualization

The new bump plot feature visualizes how tokens evolve across time for each particle:

```python
# Create bump plot visualization
fig = viz.visualize_bumplot(
    particles,
    color_by='transition_prob',  # Color by probability
    max_vocab_display=50,         # Show top 50 tokens
    show_tokens=True,              # Display token labels
    output_path='bumplot.png'
)
```

Color schemes available:
- `'transition_prob'`: Colors reflect token transition probabilities
- `'entropy'`: Colors based on entropy at each step
- `'particle_id'`: Distinct colors for each particle
