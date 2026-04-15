module PythonCallExt

using ChemBERTa
using PythonCall
const CondaPkg = PythonCall.C.CondaPkg

const chem = Ref{Py}()
  
function __init__()
    conda_env = CondaPkg.envdir()
    lib_dir = joinpath(conda_env, Sys.iswindows() ? "Lib" : "lib")
    if isdir(lib_dir)
        for entry in readdir(lib_dir)
            sp = joinpath(lib_dir, entry, "site-packages")
            isdir(sp) && pyimport("sys").path.insert(0, sp)
        end
    end

    ChemBERTa._canonicalize[] = _canonicalize_py
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
function _canonicalize_py(smiles)
    mol = _get_mol(smiles)
    string(mol) == "None" && error("Invalid SMILES: '$(smiles)'!")
    return string(_get_smiles(mol))
end

end