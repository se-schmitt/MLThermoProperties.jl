module RDKitMinimalLibExt

using ChemBERTa
using RDKitMinimalLib: RDKitMinimalLib as RDK

function ChemBERTa.canonicalize(smiles::AbstractString; is_canonical=false)
    return RDK.get_smiles(RDK.get_mol(smiles))
end

end