struct COGNACexp{c<:CL.EoSModel,T} <: COGNACModel
    components::Array{String,1}
    groups::GroupParam{T}
    params::CL.UNIFACParam{T}
    puremodel::CL.EoSVectorParam{c}
    references::Array{String,1}
    unifac_cache::CL.UNIFACCache{T}
end

function COGNACexp(components, groups, params, puremodel, references, unifac_cache)
    c = eltype(puremodel)
    T = eltype(params)
    return COGNACexp{c,T}(components, groups, params, puremodel, references, unifac_cache)
end

COGNACexp(components; kwargs...) = _cognac(COGNACexp, components; kwargs...)

"""
    COGNAC
    COGNACexp <: UNIFACModel

    COGNACexp(components;
    puremodel = BasicIdeal,
    userlocations = String[],
    pure_userlocations = String[],
    verbose = false,
    reference_state = nothing)

## Input parameters
- `SMILES`: SMILES representation of the components

## Input models
- `puremodel`: model to calculate pure pressure-dependent properties

## Description
COGNAC (local-COmposition Graph Neural network for the prediction of Activity Coefficients) model [hoffmann_learning_2026](@cite) for predicting activity coefficients (explicit version). Both `COGNAC` and `COGNACexp` can be used. The mod. UNIFAC (Dortmund) parameters ``R_a`` and ``Q_a`` (per atom) and ``a_{ab}``, ``b_{ab}``, and ``c_{ab}`` (per atom pair) are predicted by a graph neural network. The parameters are valid only for the mixture they were built for; `COGNACexp` therefore predicts them at model construction. The group activity coefficients then follow from the closed-form mod. UNIFAC equations.

## Example
```julia
using MLThermoProperties, Clapeyron

model = COGNAC(["ethanol", "water"])
model = COGNAC(["ethanol", "water"]; userlocations=(; SMILES=["CCO", "O"]))

activity_coefficient(model, 1e5, 298.15, [0.3, 0.7])
```
"""
COGNACexp

# no mask needed: out-of-block entries come out finite and `νₖᵢ = 0` drops them
cognac_g!(::COGNACexp, g, Θ, Ψ, _live) = cognac_g_explicit!(g, similar(Θ), Θ, Ψ)
