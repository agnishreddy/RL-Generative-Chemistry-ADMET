import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from rdkit.Chem import Draw
import numpy as np
import matplotlib.pyplot as plt

# Import Custom BioCore Modules
from models_rnn_architecture import SMILESGenerator, vocab_size, token2idx, idx2token, device
from oracle_reward_function import compute_reward, screen_lipinski

# Reproducibility checkpoint
torch.manual_seed(42)

def generate_and_train(generator, optimizer, epochs=1000, batch_size=32, max_len=50):
    generator.train()
    reward_history = []
    validity_history = []
    
    print(f"\n[BioCore System] Initiating De Novo RL Training on {device}...")
    
    for epoch in range(1, epochs + 1):
        hidden = generator.init_hidden(batch_size)
        input_token = torch.full((batch_size, 1), token2idx['<START>'], dtype=torch.long).to(device)
        
        batch_log_probs = torch.zeros(batch_size).to(device)
        batch_smiles = ["" for _ in range(batch_size)]
        
        # Out-of-graph list to prevent autograd corruption
        finished = [False] * batch_size 
        
        # 1. GENERATION PHASE
        for step in range(max_len):
            logits, hidden = generator(input_token, hidden)
            
            probs = F.softmax(logits[:, -1, :], dim=-1)
            dist = Categorical(probs)
            next_token = dist.sample()
            log_prob = dist.log_prob(next_token)
            
            finished_tensor = torch.tensor(finished, dtype=torch.bool).to(device)
            batch_log_probs = batch_log_probs + torch.where(finished_tensor, torch.zeros_like(log_prob), log_prob)
            
            for i in range(batch_size):
                if not finished[i]:
                    char = idx2token[next_token[i].item()]
                    if char == '<END>':
                        finished[i] = True 
                    elif char not in ['<START>', '<PAD>']:
                        batch_smiles[i] += char
            
            if all(finished):
                break
                
            input_token = next_token.unsqueeze(1)
            
        # 2. EVALUATION PHASE
        rewards = torch.tensor([compute_reward(smi) for smi in batch_smiles], dtype=torch.float32).to(device)
        
        valid_count = sum(1 for r in rewards if r > -1.0)
        avg_reward = rewards.mean().item()
        reward_history.append(avg_reward)
        validity_history.append(valid_count / batch_size)
        
        # 3. POLICY GRADIENT UPDATE
        loss = -(batch_log_probs * rewards).mean()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5.0)
        optimizer.step()
        
        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:04d} | Avg Reward: {avg_reward: .3f} | Valid SMILES: {valid_count}/{batch_size}")
            sample_idx = np.argmax(rewards.cpu().numpy())
            print(f"   -> Top Generated: {batch_smiles[sample_idx]} (Reward: {rewards[sample_idx]:.2f})")

    return reward_history, validity_history


@torch.no_grad()
def generate_virtual_library(generator, num_samples=200, max_len=50):
    """Freezes the trained agent to hallucinate a novel virtual library."""
    generator.eval()
    valid_unique_smiles = set()
    
    print(f"\n[BioCore System] Generating Virtual Library of {num_samples} candidate sequences...")
    
    batch_size = 64
    iterations = (num_samples // batch_size) + 1
    
    for _ in range(iterations):
        hidden = generator.init_hidden(batch_size)
        input_token = torch.full((batch_size, 1), token2idx['<START>'], dtype=torch.long).to(device)
        batch_smiles = ["" for _ in range(batch_size)]
        finished = [False] * batch_size
        
        for step in range(max_len):
            logits, hidden = generator(input_token, hidden)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = Categorical(probs).sample()
            
            for i in range(batch_size):
                if not finished[i]:
                    char = idx2token[next_token[i].item()]
                    if char == '<END>':
                        finished[i] = True
                    elif char not in ['<START>', '<PAD>']:
                        batch_smiles[i] += char
            
            if all(finished):
                break
            input_token = next_token.unsqueeze(1)
            
        # Add to set if valid
        for smi in batch_smiles:
            if compute_reward(smi) > -1.0: 
                valid_unique_smiles.add(smi)
                    
    print(f"[BioCore System] Virtual Library Generated. Found {len(valid_unique_smiles)} valid, unique molecules.")
    return list(valid_unique_smiles)


if __name__ == "__main__":
    # 1. Initialize Network
    generator = SMILESGenerator(vocab_size=vocab_size).to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    
    # 2. Execute Model Training
    rewards_log, valid_log = generate_and_train(generator, optimizer, epochs=600, batch_size=64)
    
    # BioCore QC: Save Weights for production deployment
    torch.save(generator.state_dict(), "generator_weights.pth")
    print("\n[BioCore System] Saved model weights to 'generator_weights.pth'")
    
    # 3. Generate Virtual Space
    virtual_library = generate_virtual_library(generator, num_samples=200)
    
    # 4. Filter and Export
    print("\n[BioCore System] Screening Virtual Library against Lipinski's Rule of Five...")
    lead_candidates, lead_mols = screen_lipinski(virtual_library)
    print(f"[BioCore System] Screening Complete. Discovered {len(lead_candidates)} highly viable Lead Candidates.")
    
    if len(lead_mols) > 0:
        sorted_leads = sorted(zip(lead_candidates, lead_mols), key=lambda x: x[0]['QED'], reverse=True)
        top_candidates = sorted_leads[:8]
        mols_to_draw = [item[1] for item in top_candidates]
        legends = [f"LogP: {item[0]['LogP']:.2f} | QED: {item[0]['QED']:.2f}" for item in top_candidates]
        
        # Save structural image to disk rather than display (better for standard Python)
        img = Draw.MolsToGridImage(mols_to_draw, molsPerRow=4, subImgSize=(300, 300), legends=legends)
        img.save("lead_candidates.png")
        print("[BioCore System] Saved lead candidates image to 'lead_candidates.png'")
