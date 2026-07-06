import torch
import torch.nn as nn

# BioCore QC: Global hardware accelerator assignment
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- SMILES VOCABULARY & EMBEDDINGS ---
# In a true SBDD pipeline, this is expanded to the ChEMBL standard vocabulary.
TOKENS = ['<PAD>', '<START>', '<END>', 'C', 'c', 'O', 'o', 'N', 'n', 'F', 'Cl', 'S', 
          '(', ')', '=', '#', '1', '2', '3', '4', '[', ']', 'H', '@', '+', '-']

vocab_size = len(TOKENS)
token2idx = {tok: i for i, tok in enumerate(TOKENS)}
idx2token = {i: tok for i, tok in enumerate(TOKENS)}

# --- THE GENERATOR (RNN) ARCHITECTURE ---
class SMILESGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=3):
        super(SMILESGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Maps integer tokens to dense continuous vectors
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Gated Recurrent Unit (GRU) learns long-term chemical syntax (e.g., closing rings)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        # Maps hidden state back to vocabulary probabilities
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        # x shape: [Batch, Sequence_Length]
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded, hidden)
        logits = self.fc(output)
        return logits, hidden
        
    def init_hidden(self, batch_size):
        # Initialize hidden state on the correct hardware device
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)