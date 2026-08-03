abstract type multHANNA_extModel <: CL.ActivityModel end

# struct multHANNAParam{T,M} <: CL.EoSParam
#     emb::SingleParam{Vector{T}}
#     scaler_T::AbstractScaler{T}
#     nn::M
#     Mw::SingleParam{T}
# end

struct multHANNA_ext{c<:CL.EoSModel,T,M} <: multHANNA_extModel
    components::Array{String,1}
    params::multHANNAParam{T,M}
    puremodel::CL.EoSVectorParam{c}
    references::Array{String,1}
end

const HANNA_ext = multHANNA_ext

"""
Description coming soon...
"""

CL.default_locations(::Type{multHANNA_ext}) = ["properties/identifiers.csv", "properties/molarmass.csv"]
get_model_path(::Type{multHANNA_ext}) = joinpath(DB_PATH, "multHANNA_ext")

function multHANNA_ext(components;
        puremodel = BasicIdeal,
        userlocations = String[],
        pure_userlocations = String[],
        verbose = false,
        reference_state = nothing,
        use_cache = true
)

    # loading SMILES und Parameter
    _components = CL.format_components(components)
    
    _params = CL.getparams(_components,CL.default_locations(multHANNA_ext);
        userlocations,ignore_headers=["dipprnumber","inchikey","cas"], ignore_missing_singleparams=["canonicalsmiles", "Mw"])

    smiles = [
        _params["canonicalsmiles"].ismissingvalues[i] ?
        ChemBERTa.canonicalize.(_params["SMILES"].values[i]) :
        _params["canonicalsmiles"].values[i]
    for i in eachindex(_components)]

    # load parameters and scalers
    ps, st = load(joinpath(get_model_path(multHANNA_ext),"parameters_states_ensemble.jld2"), "ps", "st")
    scaler_T =   load_scaler(joinpath(get_model_path(multHANNA_ext), "scaler_T.jld2"))
    scaler_emb = load_scaler(joinpath(get_model_path(multHANNA_ext), "scaler_emb.jld2"))

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
    nns = [
        multHANNA_extLux(
            theta, alpha, phi,
            ifelse(use_cache, [zeros(N_NODES,1) for _ in eachindex(_components)], nothing)
        )
        for _ in eachindex(ps)
    ]
    smodels = StatefulLuxLayer.(nns, ps, Lux.testmode.(st))

    # Precompute the Lipschitz-scaled weights once (inference reuses them)
    for smodel in smodels
        prime_scaled_weights!(smodel.ps, smodel.st)
    end

    # Calc embeddings with ChemBERTa from HuggingFace
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

    params = multHANNAParam(emb, scaler_T, smodels, _params["Mw"])
    _puremodel = CL.init_puremodel(puremodel, components, pure_userlocations, verbose)
    references = String["Reference coming soon..."] #! add references

    model = multHANNA_ext(_components, params, _puremodel, references)
    CL.set_reference_state!(model, reference_state, verbose = verbose)

    return model
end


function CL.excess_gibbs_free_energy(model::multHANNA_ext, p, T, z)
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

export HANNA_ext, multHANNA_ext
