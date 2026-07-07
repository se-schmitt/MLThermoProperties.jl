# scalers
abstract type AbstractScaler{T} end

struct Scaler{T} <: AbstractScaler{T}
    μ::T
    σ::T
end

scale(scaler::Scaler, v::T) where {T} = (v .- scaler.μ) ./ scaler.σ
unscale(scaler::Scaler, v::T) where {T} = v .* scaler.σ .+ scaler.μ

load_scaler(path::String; T=Float64) = load_scaler(path, Scaler; T)
function load_scaler(path::String, ::Type{Scaler}; T=Float64)
    @load joinpath(DB_PATH, path) μ σ
    return Scaler(T.(μ), T.(σ))
end

Base.broadcastable(scaler::AbstractScaler) = Ref(scaler)
Clapeyron.is_splittable(::AbstractScaler) = false
#TODO move to Clapeyron
Clapeyron.is_splittable(::StatefulLuxLayer) = false
Clapeyron.is_splittable(::NamedTuple) = false

function _build_es_model(components, model::ES.AbstractEntropyScalingModel; kwargs...)
    return model
end
function _build_es_model(components, model::Vector{<:ES.AbstractTransportPropertyModel}; kwargs...)
    return model
end
function _build_es_model(components, ::Type{MODEL}; kwargs...) where {MODEL<:ES.AbstractEntropyScalingModel}
    return MODEL(components; kwargs...)
end