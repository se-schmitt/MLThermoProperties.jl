import os
import torch, json
import numpy as np
from transformers import AutoModel
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from tokenizers import Regex

path_data = "../../lib/ChemBERTa/data/"

model = AutoModel.from_pretrained(
    path_data,
    local_files_only=True
)

tokenizer = Tokenizer(
    WordLevel.from_file(
        os.path.join(path_data, 'vocab.json'),
        unk_token='[UNK]'
    )
)

# Set the pre-tokenizer to split SMILES characters (including handling Br, Cl, etc.)
pre_tokenizer = Split(
    pattern=Regex(r"\[(.*?)\]|Br|Cl|."),
    behavior='isolated'
)
tokenizer.pre_tokenizer = pre_tokenizer

smiles = "CCCCO"

encoded = tokenizer.encode(smiles)
input_ids = torch.tensor([encoded.ids])
attention_mask = torch.tensor([encoded.attention_mask])
output = model(input_ids, attention_mask=attention_mask)

np.savetxt("output.csv", output.last_hidden_state[0,0,:].detach().numpy())