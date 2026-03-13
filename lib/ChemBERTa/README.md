# ChemBERTa.jl

Small sub-package for the ChemBERTa model ([Chithrananda et al. (2020)](https://arxiv.org/abs/2010.09885)) applied in the MLPROP models.

## Usage

```julia
using ChemBERTa
smiles = "CCCO"

# Loading the ChemBERTa model
bert = ChemBERTa.load()

# Calculating the embedding for a given SMILES
embedding = bert(smiles)
```

## Documentation

The ChemBERTa model is a customized model based on [ChemBERTa-77M-MTR](https://huggingface.co/DeepChem/ChemBERTa-77M-MTR) ([Ahmad et al. (2022)](https://arxiv.org/abs/2209.01712)).
The tokenizer is different from the original version as the it fixes some errors.
See [Specht et al. (2024)](https://doi.org/10.1039/D4SC05115G) for details.
