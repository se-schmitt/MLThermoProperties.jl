module PythonCallExt

using ChemBERTa
using PythonCall

const chem = Ref{Py}()
  
function __init__()
    chem[] = pyimport("rdkit.Chem")
end

# RDKit functions
function _get_mol(smiles)
    return chem[].MolFromSmiles(smiles)
end

function _get_smiles(mol)
    return chem[].MolToSmiles(mol)
end

# Canonization
function ChemBERTa.canonicalize(smiles::AbstractString; is_canonical=false)
    mol = _get_mol(smiles)
    string(mol) == "None" && error("Invalid SMILES: '$(smiles)'!")
    return string(_get_smiles(mol))
end

end