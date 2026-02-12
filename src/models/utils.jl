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

# Show method
function Base.show(io::IO, mime::MIME"text/plain", model::MLPROP_MODELS)
    print(io, nameof(typeof(model)))
    length(model) == 1 && println(io, " with 1 component:")
    length(model) > 1 && println(io, " with ", length(model), " components:")
    CL.show_pairs(io,CL.component_list(model))
    
    CL.show_info(io,model)
    CL.show_params(io,model)
    CL.show_reference_state(io,model)
    CL.may_show_references(io,model)
end