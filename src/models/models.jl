include("utils.jl")

include("grappa.jl")
include("HANNA/hanna.jl")

MLPROP_MODELS = Union{GRAPPA, ogHANNA}