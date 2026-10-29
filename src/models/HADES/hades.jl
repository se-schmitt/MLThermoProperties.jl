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
    vismodel::M
    references::Array{String,1}
end


"""
    HADES <: AbstractTransportPropertyModel

    HADES(components;
    vismodel = RefpropRES,
    userlocations = String[],
    vis_userlocations = String[],
    verbose = false)

## Input parameters
- `SMILES`: canonical SMILES (using RDKit) representation of the components
- `Mw`: single parameter (`Float64`) (Optional) - Molecular Weight `[g·mol⁻¹]`

## Input models
- `vismodel`: viscosity model, either a vector of pure component viscosity models or a mixture viscosity model

## Description

Hybrid Architecture for consistent Diffusion-coefficient Estimation across Solvents (HADES) [wagner_comp_2026](@cite).
`HADES` predicts self-diffusion coefficients in liquid mixtures and is applicable to organic components (molar mass < 1000 g·mol⁻¹, no atoms heavier than chlorine) and water.
It combines the Stokes-Einstein equation with a deep-set neural network.

The self-diffusion coefficients of all components can be calculated by calling [`self_diffusion_coefficient`](https://se-schmitt.github.io/EntropyScaling.jl/dev/).

The viscosity of the mixture ``\\eta_\\mathrm{solv}`` is obtained from `vismodel` (default: `RefpropRES`):
- If a vector of pure component viscosity models is given, the pure component viscosities are mixed logarithmically, ``\\ln \\eta_\\mathrm{solv} = \\sum_k x_k \\ln \\eta_k``, as done for the training of `HADES` (recommended).
- If a mixture viscosity model is given, its viscosity is used directly.

## Example
```julia
using MLThermoProperties, EntropyScaling

components = ["ethanol", "water", "methanol"]
smiles = ["CCO", "O", "CO"]
vismodel = [ConstantModel(Viscosity(), η) for η in [0.00104, 0.000891, 0.000543]]   # pure component viscosities at 298.15 K in Pa·s

model = HADES(components; userlocations=(;SMILES=smiles), vismodel)
D_self = self_diffusion_coefficient(model, 1e5, 298.15, [0.2, 0.4, 0.4])   # in m²/s
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
    
    _params = CL.getparams(components,CL.default_locations(HADES);
        userlocations,ignore_headers=["dipprnumber","inchikey","cas","canonicalsmiles"],
        ignore_missing_singleparams=["Mw"])

    smiles = _params["SMILES"].values

    ps, st = load(joinpath(get_model_path(HADES),"parameters_states_ensemble.jld2"), "ps", "st")

    smodels = StatefulLuxLayer.(Ref(HADESLux()), ps, Lux.testmode.(st))

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
    # _pure_vismodels = _split_es_model(_vismodel)
    references = String["10.48550/arXiv.2609.24599"]

    return HADES(_components, params, _vismodel, references)
end

Base.broadcastable(x::HADES) = Ref(x)

# mixture viscosity model given -> use directly
_mixture_viscosity(vismodel, p, T, z; phase) = viscosity(vismodel, p, T, z; phase)

# pure component viscosity models given -> logarithmic mixing 
function _mixture_viscosity(vismodels::AbstractVector, p, T, z; phase)
    return exp(sum(z[k] * log(viscosity(vismodels[k], p, T; phase)) for k in eachindex(z)))
end

# gets all self diffusion coefficients of the mixture
function ES.self_diffusion_coefficient(model::HADES, p, T, z; phase=:liquid)
    N_comps = length(model.components)
    # Initialitzing constants required for Stokes-Einstein-equation
    f = 0.64
    ϱ_ref = 1050.
    
    η_mix = _mixture_viscosity(model.vismodel, p, T, z; phase)
    TT = Base.promote_eltype(p,T,z,η_mix)
    D_self = zeros(TT, N_comps)

    for idx_i in 1:N_comps
        M_i = model.params.Mw.values[idx_i] * 1e-3
        
        r_i = cbrt(f * 3 * M_i / (4π * ϱ_ref * NA))
        Dij_SE = (kB * T) / (6π * η_mix * r_i) 
        
        _X_ = circshift(model.params.Xs, (0, -idx_i + 1))
        _z_ = circshift(z, -idx_i + 1)

        _X = reshape(_X_, size(_X_, 1), size(_X_, 2), 1)
        _z = reshape(_z_, 1, length(_z_), 1)
        smask = ones(1, N_comps, 1) 

        cf_sum = zero(TT)
        for smodel in model.params.smodels
            cf_sum += first(smodel((_X, _z, smask)))
        end
        cf_mean = cf_sum / length(model.params.smodels)

        D_self[idx_i] = Dij_SE * cf_mean
    end

    return D_self
end

export HADES