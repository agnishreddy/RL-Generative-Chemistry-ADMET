 🧬 End-to-End AI Drug Discovery: De Novo Generation & ADMET Profiling

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-blue?style=for-the-badge)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Reinforcement_Learning-purple?style=for-the-badge)

 📌 Overview
This repository contains an enterprise-grade computational pipeline for **Hit-to-Lead Optimization** and **De Novo Generative Chemistry**. It bridges graph-based deep learning (for property prediction) with reinforcement learning (for molecular generation) to hallucinate novel, highly bioavailable chemical structures while profiling their ADMET (Absorption, Distribution, Metabolism, Excretion, and Toxicity) viability.

 🏗️ Architecture

 1. Generative Agent: Recurrent Neural Network (RNN)
- **Sequence Generation:** A deep Gated Recurrent Unit (GRU) engineered to learn the granular syntax of SMILES strings, operating character-by-character to generate novel chemical space.
- **Optimization Strategy:** The generator is optimized via **REINFORCE Policy Gradients**, shifting the network's weights from random character generation to valid, targeted molecular design based on environmental rewards.

 2. Discriminative Oracle: Multi-Task Graph Attention Network (GAT)
- **Shared Representation:** Utilizes a PyTorch Geometric Graph Attention Network (`GATConv`) to ingest 2D molecular topologies (atoms as nodes, bonds as edges).
- **Multi-Task Learning:** Simultaneously predicts **Aqueous Solubility (ESOL)** via regression and **12 Pathways of Toxicity (Tox21)** via binary classification.

 ⚖️ The Oracle: Multi-Parameter Optimization (MPO)
Unconstrained generation often leads to mode collapse or chemically impossible structures. To prevent this, the RNN is encapsulated in a Reinforcement Learning loop governed by a biochemical Oracle.

The reward function performs **Multi-Parameter Optimization (MPO)** utilizing **RDKit** to evaluate:
- **Chemical Validity:** Strict penalties for broken valences or unclosed rings.
- **Quantitative Estimate of Drug-likeness (QED):** Directly calculated to ensure structural drug-likeness.
- **Crippen LogP:** Penalizes deviation from the ideal lipophilicity threshold (LogP ~ 2.0), acting as a strict proxy for **ADMET viability** and oral absorption.

Generated virtual libraries are subsequently filtered through **Lipinski's Rule of Five** to identify viable Lead Candidates.

 🛠️ Engineering Bottlenecks Solved
Building this pipeline required solving several critical gradient and computational graph bottlenecks:

1. **PyTorch Autograd Inplace Modification Corruption:**
   - *Issue:* During the RL rollout phase, boolean masking tensors used to track sequence termination triggered inplace modification errors (`RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation`), destroying the computational graph prior to `.backward()`.
   - *Resolution:* Explicitly resolved this by decoupling the tracking logic using out-of-graph Python lists and engineering out-of-place tensor addition to preserve flawless gradient flow during REINFORCE backpropagation.

2. **Sparse Data Masking (NaN Propagation):**
   - *Issue:* Real-world assay datasets like Tox21 are highly sparse (missing tests encoded as `NaN`). Standard loss evaluation against `NaN` targets mathematically corrupted the entire batch gradient (`NaN * 0 = NaN`).
   - *Resolution:* Engineered safe NaN-masking for sparse multi-task learning utilizing `torch.where` to sanitize targets with zeros *before* loss computation, combined with epsilon normalizations to prevent zero-division.

 🚀 Quick Start

 1. Install Dependencies
```bash
pip install torch rdkit torch-geometric matplotlib numpy
```

 2. Execute the Reinforcement Learning Pipeline
Initiate the REINFORCE loop to train the generative agent. The script will automatically generate a virtual library, screen it, and output the viable Lead Candidates.
```bash
python train_rl.py
```
*Outputs: Model weights saved to `generator_weights.pth` and a 2D structural grid of top candidates exported to `lead_candidates.png`.*
