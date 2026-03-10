include("grappa.jl")
include("ese.jl")

# --- utils ---
# placeholder function for RDKit
function get_descriptors(smiles)
    error("""
    To use this functionality in `MLPROP.jl`, you need to install and import either `PythonCall.jl` or `RDKitMinimalLib.jl`!
    """)
end
