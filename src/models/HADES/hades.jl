include("layers_hades.jl")

abstract type HADESModel <: ES.AbstractTransportPropertyModel end

@concrete struct HADESParam{T, S}
    Xs::Matrix{T}
    Mw::SingleParam{T}
    smodels::Vector{S}
end

struct HADES{T,M,S} <: HADESModel
    components::Array{String,1}
    params::HADESParam{T, S}
    pure_vismodels::Vector{M}
    references::Array{String,1}
end

"""
    HADES <: AbstractTransportPropertyModel

    HADES(components;
    vismodel = nothing,
    userlocations = String[],
    vis_userlocations = String[],
    verbose = false)

## Input parameters
- `SMILES`: canonical SMILES (using RDKit) representation of the components
- `Mw`: single parameter (`Float64`) (Optional) - Molecular Weight `[g·mol⁻¹]`
- `vismodel`: viscosity model 

## Description

Description to come...

## Examples
```julia
using MLThermoProperties, EntropyScaling

model = HADES(["ethanol", "acetonitrile"])
D_matrix = inf_diffusion_coefficient(model, 1e5, 300.)
D_eth = inf_diffusion_coefficient(model, 1e5, 300.; solute="ethanol", solvent="acetonitrile")
```
"""
HADES

CL.default_locations(::Type{HADES}) = ["properties/identifiers.csv"]
get_model_path(::Type{HADES}) = joinpath(DB_PATH, "HADES")

function HADES(components;
        vismodel = RefpropRES,
        userlocations = String[],
        vis_userlocations = String[],
        verbose = false,
)

    # loading SMILES und Parameter
    _components = CL.format_components(components)
    N_comps = length(_components)
    
    _params = CL.getparams(components,CL.default_locations(HADES);
        userlocations,ignore_headers=["dipprnumber","inchikey","cas","canonicalsmiles"],
        ignore_missing_singleparams=["Mw"])

    smiles = _params["SMILES"].values

    nn = HADESLux()
    ps, st = load(joinpath(get_model_path(HADES),"parameters_states_ensemble.jld2"), "ps", "st")

    nns = [HADESLux() for _ in eachindex(ps)]
    smodels = StatefulLuxLayer.(nns, ps, Lux.testmode.(st))

    # get chem data
    X = get_ese_X.(smiles)
    Xs = hcat(X...)

    Mw = _params["Mw"]
    for i in eachindex(_components)
        Mw.ismissingvalues[i] || continue
        Mw.values[i] = Xs[1,i]*1e3
        Mw.ismissingvalues[i] = false
    end

    params = HADESParam(Xs, Mw, smodels)
    _vismodel = _build_es_model(_components, vismodel; userlocations=vis_userlocations)
    _pure_vismodels = _split_es_model(_vismodel)
    references = String[""]

    return HADES(_components, params, _pure_vismodels, references)
end


Base.broadcastable(x::HADES) = Ref(x)

# index-based self diffusion coefficient 
function ES.self_diffusion_coefficient(model::HADES, p, T, z, idx_i; phase=:liquid)
    TT = Base.promote_eltype(p,T)

    # Initialitzing constants required for Stokes-Einstein-equation
    f = 0.64
    ϱ_ref = 1050.
    M_i = model.params.Mw.values[idx_i]*1e-3
    
    η_pures = [viscosity(vismodel, p, T; phase) for vismodel in model.pure_vismodels]
    η_mix = exp(sum(z .* log.(η_pures)))

    r_i = cbrt(f * 3 * M_i / (4π * ϱ_ref * NA))
    Dij_SE = (kB*T)/(6π*η_mix*r_i)  # Stokes-Einstein-equation

    _X_ = circshift(model.params.Xs, (0, -idx_i + 1))
    _z_ = circshift(z, -idx_i + 1)

    _X = reshape(_X_, size(_X_, 1), size(_X_, 2), 1)
    _z = reshape(_z_, 1, length(_z_), 1)
    smask = ones(Float32, 1, length(z))  

    # mean
    nn_outs = [smodel((_X, _z, smask)) for smodel in model.params.smodels]
    cfs = [first(out) for out in nn_outs]
    cf_mean = sum(cfs) / length(cfs)    

    return Dij_SE * cf_mean
end

# gets all self diffusion coefficients of the mixture
function ES.self_diffusion_coefficient(model::HADES, p, T, z; phase=:liquid)
    TT = Base.promote_eltype(p,T)
    N_comps = length(model.components)
    # Initialitzing constants required for Stokes-Einstein-equation
    f = 0.64
    ϱ_ref = 1050.
    
    η_pures = [viscosity(vismodel, p, T; phase) for vismodel in model.pure_vismodels]
    η_mix = exp(sum(z .* log.(η_pures)))

    D_self = zeros(typeof(η_mix), N_comps)

    for idx_i in 1:N_comps
        M_i = model.params.Mw.values[idx_i] * 1e-3
        
        r_i = cbrt(f * 3 * M_i / (4π * ϱ_ref * NA))
        Dij_SE = (kB * T) / (6π * η_mix * r_i) 
        
        _X_ = circshift(model.params.Xs, (0, -idx_i + 1))
        _z_ = circshift(z, -idx_i + 1)

        _X = reshape(_X_, size(_X_, 1), size(_X_, 2), 1)
        _z = reshape(_z_, 1, length(_z_), 1)
        smask = ones(1, N_comps, 1) 

        nn_outs = [smodel((_X, _z, smask)) for smodel in model.params.smodels]
        cfs = [first(out) for out in nn_outs]
        cf_mean = sum(cfs) / length(cfs)    

        D_self[idx_i] = Dij_SE * cf_mean
    end

    return D_self
end

export HADES