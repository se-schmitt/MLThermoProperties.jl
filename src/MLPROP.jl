module MLPROP

using Clapeyron, Lux, ConcreteStructs, ChemBERTa, JLD2, EntropyScaling

const CL = Clapeyron
const ES = EntropyScaling

const DB_PATH = normpath(Base.pkgdir(MLPROP),"database")

include("models/models.jl")

end
