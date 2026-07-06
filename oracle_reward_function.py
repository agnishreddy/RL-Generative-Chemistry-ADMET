from rdkit import Chem
from rdkit.Chem import QED, Descriptors
from rdkit.Chem.Descriptors import MolLogP

# --- THE ORACLE (REWARD FUNCTION) ---
def compute_reward(smiles):
    """
    BioCore MPO (Multi-Parameter Optimization) Oracle.
    Rewards valid chemistry, high QED (drug-likeness), and optimal LogP.
    """
    mol = Chem.MolFromSmiles(smiles)
    
    # Bottleneck 1: Invalid Chemistry -> Severe Penalty
    if mol is None:
        return -2.0 
    
    try:
        # Sanitize molecule (check valence, aromaticity)
        Chem.SanitizeMol(mol)
        
        # Bottleneck 2: Descriptors
        qed = QED.qed(mol)          # Range: 0.0 to 1.0
        logp = MolLogP(mol)         # Proxy for solubility/lipophilicity
        
        # We want LogP to be around 2.0 (optimal oral ADMET)
        logp_penalty = abs(logp - 2.0)
        
        # Total Reward: Prioritize valid drug-like structures in the LogP sweet spot
        reward = (qed * 2.0) - logp_penalty + 1.0
        return reward
        
    except Exception:
        # Fails valence checks
        return -1.0

# --- ADMET PROFILING (LIPINSKI'S RULE OF 5) ---
def screen_lipinski(virtual_library):
    """
    Screens an in silico generated library against Lipinski's constraints.
    Returns lead candidates and their RDKit molecule objects.
    """
    lead_candidates = []
    lead_mols = []

    for smi in virtual_library:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
            
        # Calculate ADMET proxies
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        tpsa = Descriptors.TPSA(mol)
        qed_score = QED.qed(mol)
        
        # Lipinski rigid filters (The "Rule of Five")
        if (mw <= 500) and (logp <= 5.0) and (hbd <= 5) and (hba <= 10):
            # Additional Pharma constraint: High drug-likeness (QED > 0.5)
            if qed_score > 0.5:
                lead_candidates.append({
                    'smiles': smi,
                    'MW': mw,
                    'LogP': logp,
                    'HBD': hbd,
                    'HBA': hba,
                    'TPSA': tpsa,
                    'QED': qed_score
                })
                lead_mols.append(mol)
                
    return lead_candidates, lead_mols