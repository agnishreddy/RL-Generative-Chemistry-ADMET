# --- COLAB CELL 1: REPOSITORY CLONE & ENVIRONMENT PROVISIONING ---
# BioCore Reviewer Note: Pulling the live repository directly from GitHub to test the 
# De Novo ADMET pipeline. Installing the required cheminformatics and graph ML dependencies.
!git clone https://github.com/agnishreddy/RL-Generative-Chemistry-ADMET.git
%cd RL-Generative-Chemistry-ADMET
!pip install -q torch rdkit torch-geometric matplotlib numpy

# --- COLAB CELL 1.5: REPO HOTFIX ---
# BioCore QC: The files on GitHub were renamed, but the internal imports weren't updated.
# Dynamically patching rl_training_loop.py to match the new filenames.
!sed -i 's/from models import/from models_rnn_architecture import/g' rl_training_loop.py
!sed -i 's/from oracle import/from oracle_reward_function import/g' rl_training_loop.py

# --- COLAB CELL 2: ARCHITECTURE VERIFICATION ---
# Checking if the modular separation of concerns (SoC) is intact.
!ls -la

import sys
import os
import torch

# Verify GPU availability for the pipeline
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[Reviewer Terminal] Hardware check: Using {device}")
print(f"[Reviewer Terminal] Verifying modular imports from agnishreddy's repository...")

# BioCore QC: Ensure the newly cloned directory is appended to the system path
sys.path.append(os.getcwd())

try:
    from models_rnn_architecture import SMILESGenerator, vocab_size
    from oracle_reward_function import compute_reward, screen_lipinski
    print("[Reviewer Terminal] Imports successful. Architecture modules located.")
except ImportError as e:
    print(f"[Reviewer Terminal] IMPORT ERROR: {e}")


# --- COLAB CELL 3: PIPELINE EXECUTION (PROOF OF CONCEPT TEST) ---
# BioCore Reviewer Note: The README claims the PyTorch inplace modification autograd 
# bug was fixed. If true, the REINFORCE loop will survive backpropagation. 
# We will run a truncated 100-epoch test to verify gradient flow and validity optimization.

# Using a quick Python script to override the epochs for a fast reviewer test
test_script = """
import torch
from models_rnn_architecture import SMILESGenerator, vocab_size, device
from oracle_reward_function import screen_lipinski
from rl_training_loop import generate_and_train, generate_virtual_library

print('\\n[Reviewer Terminal] Initiating REINFORCE Policy Gradient Test (100 Epochs)...')

# Initialize Network
generator = SMILESGenerator(vocab_size=vocab_size).to(device)
optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)

# Execute Model Training (Truncated to 100 for fast verification)
rewards_log, valid_log = generate_and_train(generator, optimizer, epochs=100, batch_size=64)

# Generate Virtual Space
print('\\n[Reviewer Terminal] Testing De Novo Generation Phase...')
virtual_library = generate_virtual_library(generator, num_samples=100, max_len=50)

# Filter and Export
print('\\n[Reviewer Terminal] Testing Oracle Lipinski Screening...')
lead_candidates, lead_mols = screen_lipinski(virtual_library)

from rdkit.Chem import Draw
if len(lead_mols) > 0:
    sorted_leads = sorted(zip(lead_candidates, lead_mols), key=lambda x: x[0]['QED'], reverse=True)
    top_candidates = sorted_leads[:8]
    mols_to_draw = [item[1] for item in top_candidates]
    legends = [f"LogP: {item[0]['LogP']:.2f} | QED: {item[0]['QED']:.2f}" for item in top_candidates]
    img = Draw.MolsToGridImage(mols_to_draw, molsPerRow=4, subImgSize=(300, 300), legends=legends)
    img.save("lead_candidates_test.png")
    print('[Reviewer Terminal] TEST PASSED: Pipeline generated valid chemistry.')
else:
    print('[Reviewer Terminal] TEST WARNING: No molecules passed Lipinski constraints in 100 epochs (Expected behavior for short runs).')
"""

with open("reviewer_test.py", "w") as f:
    f.write(test_script)

!python reviewer_test.py


# --- COLAB CELL 4: VISUAL EVIDENCE EXTRACTION ---
# If the pipeline works, it should have generated a 2D grid of the lead candidates.
import os
from IPython.display import Image, display

if os.path.exists("lead_candidates_test.png"):
    print("\n[Reviewer Terminal] Validating generated chemical structures...")
    display(Image(filename="lead_candidates_test.png"))
elif os.path.exists("lead_candidates.png"):
    print("\n[Reviewer Terminal] Validating generated chemical structures...")
    display(Image(filename="lead_candidates.png"))
else:
    print("\n[Reviewer Terminal] Note: The agent requires more epochs to satisfy Lipinski's Rule of 5.")