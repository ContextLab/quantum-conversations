"""
Execute notebook cells to generate figures.
"""

import os
import sys
sys.path.append('.')

# Change to notebooks directory for proper paths
os.chdir('notebooks')

# Execute the notebook code
exec(open('quantum_conversations_demo.ipynb', 'r').read())

print("Notebook execution attempted.")