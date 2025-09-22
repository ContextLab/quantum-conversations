"""
Script to run the quantum conversations notebook and generate visualizations.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Run the notebook
print("Running quantum conversations notebook...")
os.system("cd code && jupyter nbconvert --to notebook --execute quantum_conversations_demo.ipynb --output quantum_conversations_demo_executed.ipynb")

print("\nNotebook execution complete! Check quantum_conversations_demo_executed.ipynb for results.")
print("Visualizations should be saved in data/derivatives/particle_visualizations/")