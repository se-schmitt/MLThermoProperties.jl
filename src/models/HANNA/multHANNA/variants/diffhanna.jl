abstract type diffHANNAModel <: multHANNAModel end

struct diffHANNA{c<:CL.EoSModel,T,M} <: diffHANNAModel
    components::Array{String,1}
    params::multHANNAParam{T,M}
    puremodel::CL.EoSVectorParam{c}
    references::Array{String,1}
end

CL.default_locations(::Type{diffHANNA}) = ["properties/identifiers.csv", "properties/molarmass.csv"]
get_model_path(::Type{diffHANNA}) = joinpath(DB_PATH, "diffHANNA")

function diffHANNA(components;
        puremodel = BasicIdeal,
        userlocations = String[],
        pure_userlocations = String[],
        verbose = false,
        reference_state = nothing,
)
    return _build_multhanna(
        diffHANNA, components; 
        puremodel, userlocations, pure_userlocations, verbose, reference_state
    )
end

# helper functions
function _build_multhanna_lux(::Type{diffHANNA}, theta, alpha, phi)
    return diffHANNALux(theta, alpha, phi)
end

function _build_multhanna_param(::Type{diffHANNA}, emb, θs, scaler_T, smodels, _params)
    return multHANNAParam(emb, θs, scaler_T, smodels, _params["Mw"])
end

# Lux layer
@concrete struct diffHANNALux <: AbstractMultHANNALux{(:theta, :alpha, :phi)}
    theta
    alpha
    phi
end

# similarity
function calc_similarity!(similarity, model::diffHANNALux, θs)
    N = length(θs)
    for i in 1:N, j in (i+1):N
        sim = cosine_similarity(θs[i], θs[j])
        similarity[i,j] = sim
        similarity[j,i] = sim
    end
end
