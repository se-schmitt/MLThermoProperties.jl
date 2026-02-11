import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import types  

current_script_path = os.path.dirname(os.path.abspath(__file__))
dependency_folder = os.path.join(current_script_path, "Python_HANNA_legacy")
if dependency_folder not in sys.path:
    sys.path.append(dependency_folder)

sys.modules['utils'] = types.ModuleType('utils')
from chemberta_utils import initialize_ChemBERTA
import Own_Scaler
sys.modules['utils.Own_Scaler'] = Own_Scaler
from Utils import predict, create_embedding_matrix

# HANNA Model
class HANNA(nn.Module):
    def __init__(self, Embedding_ChemBERT=384, nodes=96):
        super(HANNA, self).__init__()
        self.Embedding_ChemBERT = Embedding_ChemBERT 
        self.nodes = nodes 
        self.theta = nn.Sequential(nn.Linear(Embedding_ChemBERT, nodes), nn.SiLU())
        self.alpha = nn.Sequential(nn.Linear(nodes+2, nodes), nn.SiLU(), nn.Linear(nodes, nodes), nn.SiLU())
        self.phi = nn.Sequential(nn.Linear(nodes, nodes), nn.SiLU(), nn.Linear(nodes, 1))

    def forward(self, temperature, mole_fractions, E_i):
        # Determine batch_size (B) and number of components (N)
        batch_size, num_components, _ = E_i.shape # [B,N,E] E=384, ChemBERTa-2 embedding

        # Enable gradient tracking to use autograd
        E_i.requires_grad_(True)
        temperature.requires_grad_(True) # Standardized temperature
        mole_fractions.requires_grad_(True) # x_1

        # Calculate remaining mole fraction for the Nth component (here: N=2)
        mole_fraction_N = 1 - mole_fractions.sum(dim=1, keepdim=True) # x_2=1-x_1 [B,1]
        mole_fractions_complete = torch.cat([mole_fractions, mole_fraction_N], dim=1) # [x_1,1-x_1], [B,2]

        # Reshape mole fraction and temperature
        mole_fractions_complete_reshaped = mole_fractions_complete.unsqueeze(-1) # [B,N,1]
        T_reshaped = temperature.view(batch_size, 1, 1).expand(-1, num_components, 1) # [B,N,1]

        # Fine-tuning of the component embeddings
        theta_E_i = self.theta(E_i) # [B,N,nodes]

        # Calculate cosine similarity between the two components
        cosine_sim = F.cosine_similarity(theta_E_i[:, 0, :], theta_E_i[:, 1, :], dim=1) #[B]
        # Calculate cosine distance between the two components
        cosine_distance = 1 - cosine_sim # [B]

        # Concatenate embedding with T and x_i
        c_i = torch.cat([T_reshaped, mole_fractions_complete_reshaped, theta_E_i], dim=-1) #[B,N,nodes+2]
        alpha_c_i = self.alpha(c_i) # [B,N,nodes]
        c_mix = alpha_c_i.sum(dim=1) # [B,nodes]
        gE_NN = self.phi(c_mix).squeeze(-1) # [B]

        # Apply cosine similarity adjustment
        correction_factor_mole_fraction = torch.prod(mole_fractions_complete, dim=1) # [B] x1*(1-x1) term
        gE = gE_NN * correction_factor_mole_fraction * cosine_distance  # [B] Adjust gE_NN with the physical constraints and calculate gE/RT

        # Compute (dgE/dx1)/RT
        dgE_dx1 = torch.autograd.grad(gE.sum(), mole_fractions, create_graph=True)[0] # [B,1]

        # ln gamma_i equation (binary mixture). Unsqueeze to adjust dimension to [B,1] for gE/RT
        ln_gamma_1 = gE.unsqueeze(1) + (1 - mole_fractions) * dgE_dx1 # [B,1]
        ln_gamma_2 = gE.unsqueeze(1) - mole_fractions * dgE_dx1 # [B,1]
        # Concatenate the ln_gammas
        ln_gammas = torch.cat([ln_gamma_1, ln_gamma_2], dim=1) # [B,2]

        return ln_gammas, gE

# Paths for model and scaler
current_script_folder = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(current_script_folder, "Python_HANNA_legacy")

model_path = os.path.join(data_folder, "HANNA_legacy_Val.pt")
scaler_path = os.path.join(data_folder, "scalerHANNA_legacy_Val.pkl")
# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HANNA().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
# Set the model to evaluation mode
model.eval()
# Load the scaler
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Initialize ChemBERTa
ChemBERTA, tokenizer = initialize_ChemBERTA(model_name="DeepChem/ChemBERTa-77M-MTR", device=None)

# Example usage with ethanol and water at 300 K:
SMILES_1 = "O" # SMILES of component 1 (water)
SMILES_2 = "CCO" # SMILES of component 2 (ethanol)
T = 300 # Temperature in K
x1_values = np.array([0.5]) # Mole fraction(s) of component 1

embedding_matrix =create_embedding_matrix(SMILES_1, SMILES_2, T, device, ChemBERTA, tokenizer, x1_values) # Create the embedding matrix
x_pred, ln_gammas_pred = predict(embedding_matrix, scaler, model, device) # Predict the logarithmic activity coefficients

# Results
ln_g1 = ln_gammas_pred[0, 0] # First Component
ln_g2 = ln_gammas_pred[0, 1] # Second Component

g1 = np.exp(ln_g1)
g2 = np.exp(ln_g2)

print(f"REFERENCE (PYTHON ORIGINAL (HANNA_LEGACY)):")
print(f"Gamma First Comp.:  {g1:.6f}")
print(f"Gamma Second Comp: {g2:.6f}")
