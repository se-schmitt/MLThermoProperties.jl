abstract type multHANNAModel <: AbstractMultHANNA end

struct multHANNAParam{T,M} <: CL.EoSParam
    emb::SingleParam{Vector{T}}
    scaler_T::AbstractScaler{T}
    nn::M
    Mw::SingleParam{T}
end

struct multHANNA{c<:CL.EoSModel,T,M} <: multHANNAModel
    components::Array{String,1}
    params::multHANNAParam{T,M}
    puremodel::CL.EoSVectorParam{c}
    references::Array{String,1}
end

const HANNA = multHANNA

"""
    HANNA <: ActivityModel
    multHANNA

    HANNA(components;
    puremodel = nothing,
    userlocations = String[],
    pure_userlocations = String[],
    verbose = false,
    reference_state = nothing)

## Input parameters
- `SMILES`: canonical SMILES (using RDKit) representation of the components
- `Mw`: Single Parameter (`Float64`) (Optional) - Molecular Weight `[g·mol⁻¹]`

## Input models
- `puremodel`: model to calculate pure component properties

## Description
Hard-Constraint Neural Network for Consistent Activity Coefficient Prediction (HANNA).
`HANNA` was trained on all available binary VLE data (up to 10 bar) and limiting activity coefficients from the Dortmund Data Bank.

## Example
```julia
using MLThermoProperties, Clapeyron

components = ["dmso", "ethanol", "aspirin"]
Mw = [78.13, 46.068, 180.158]
smiles = ["CS(=O)C", "CCO", "CC(=O)Oc1ccccc1C(=O)O"]

model = HANNA(components,userlocations=(;Mw=Mw, SMILES=smiles))
# model = HANNA(components) # also works if components are in the database 
```

## References
1.  M. Hoffmann, T. Specht, Q. Göttl, J. Burger, S. Mandt, H. Hasse, and F. Jirasek: A Machine-Learned Expression for the Excess Gibbs Energy, (2025), DOI: https://doi.org/10.48550/arXiv.2509.06484.
"""
multHANNA

CL.default_locations(::Type{multHANNA}) = ["properties/identifiers.csv", "properties/molarmass.csv"]
get_model_path(::Type{multHANNA}) = joinpath(DB_PATH, "multHANNA")

function multHANNA(components;
        puremodel = BasicIdeal,
        userlocations = String[],
        pure_userlocations = String[],
        verbose = false,
        reference_state = nothing,
        use_cache = true
)
    return _build_multhanna(
        multHANNA, components; 
        puremodel, userlocations, pure_userlocations, verbose, reference_state, use_cache
    )
end

# Build a multHANNA model
function _build_multhanna(
    MODEL, components; 
    puremodel, userlocations, pure_userlocations, verbose, reference_state, use_cache
)
    # loading SMILES und Parameter
    _components = CL.format_components(components)
    
    _params = CL.getparams(_components,CL.default_locations(MODEL);
        userlocations,ignore_headers=["dipprnumber","inchikey","cas"], ignore_missing_singleparams=["canonicalsmiles", "Mw"])

    smiles = [
        _params["canonicalsmiles"].ismissingvalues[i] ?
        ChemBERTa.canonicalize.(_params["SMILES"].values[i]) :
        _params["canonicalsmiles"].values[i]
    for i in eachindex(_components)]

    # load parameters and scalers
    ps, st = load(joinpath(get_model_path(MODEL),"parameters_states_ensemble.jld2"), "ps", "st")
    scaler_T =   load_scaler(joinpath(get_model_path(MODEL), "scaler_T.jld2"))
    scaler_emb = load_scaler(joinpath(get_model_path(MODEL), "scaler_emb.jld2"))

    # Create model
    N_EMB = 384
    N_NODES = 96
    
    theta = LipschitzDense(N_EMB, N_NODES, silu)
    alpha = Chain(
        LipschitzDense(N_NODES + 2, N_NODES, silu),
        LipschitzDense(N_NODES, N_NODES, silu)
    )
    phi = Chain(
        LipschitzDense(N_NODES, N_NODES, silu),
        LipschitzDense(N_NODES, 1, identity)
    )    
    nns = [_build_multhanna_lux(MODEL, theta, alpha, phi, _components) for _ in eachindex(ps)]
    smodels = StatefulLuxLayer.(nns, ps, Lux.testmode.(st))

    # Precompute the Lipschitz-scaled weights once (inference reuses them)
    for smodel in smodels
        prime_scaled_weights!(smodel.ps, smodel.st)
    end

    # Calc embeddings
    if isnothing(BERT)
        global BERT = ChemBERTa.load()
    end
    emb = SingleParam("ChemBERTa embedding", _components, scale.(scaler_emb, BERT.(smiles; is_canonical=true)))

    # Set θ caches
    if use_cache
        for smodel in smodels, i in eachindex(_components)
            smodel.model.__cache_θs[i] .= first(smodel.model.theta(emb[i], smodel.ps.theta, smodel.st.theta))
        end
    end

    params = _build_multhanna_param(MODEL, emb, scaler_T, smodels, _params)
    _puremodel = CL.init_puremodel(puremodel, components, pure_userlocations, verbose)
    references = String["10.48550/arXiv.2509.06484"]

    model = MODEL(_components, params, _puremodel, references)
    CL.set_reference_state!(model, reference_state, verbose = verbose)

    return model
end

# helper functions
function _build_multhanna_lux(::multHANNA, theta, alpha, phi, c)
    _cache = ifelse(use_cache, [zeros(N_NODES,1) for _ in eachindex(c)], nothing)
    return multHANNALux(theta, alpha, phi, _cache, 100.0)
end

function _build_multhanna_param(::multHANNA, emb, scaler_T, smodels, _params)
    return multHANNAParam(emb, scaler_T, smodels, _params["Mw"])
end

# gE
function CL.excess_gibbs_free_energy(model::AbstractMultHANNA, p, T, z)
    x = z ./ sum(z) 
    
    params = model.params
    # Embeddings and RBF-Gamma
    embs = params.emb.values
    
    T_scaled = scale(params.scaler_T, T) 
    
    # loop over all ensemble models
    gE_sum = zero(eltype(x)) 
    num_models = length(model.params.nn)
    
    for nn in model.params.nn
        gE_sum += nn((T_scaled, x, embs),)
    end
    
    gE_mean_dim_less = gE_sum / num_models
    
    return gE_mean_dim_less * Rgas(model) * T * sum(z)
end

export HANNA, multHANNA
