@concrete struct LipschitzDense <: LuxCore.AbstractLuxLayer
    activation
    in_dims<:Lux.IntegerType
    out_dims<:Lux.IntegerType
    init_weight
    init_bias
    init_ci
    eps::Real
end

function Base.show(io::IO, d::LipschitzDense)
    print(io, "LipschitzDense($(d.in_dims) => $(d.out_dims)")
    (d.activation == identity) || print(io, ", $(d.activation)")
    return print(io, ")")
end

function LipschitzDense(in_dims::Integer, out_dims::Integer, act; 
        init_weight=glorot_uniform, init_bias=zeros32, init_ci=rng->rand(rng,Float32,1)*20, eps=1f-12)
    return LipschitzDense(act, in_dims, out_dims, init_weight, init_bias, init_ci, eps)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::LipschitzDense)
    return (;
        weight = l.init_weight(rng, l.out_dims, l.in_dims), 
        bias = l.init_bias(rng, l.out_dims),
        ci = l.init_ci(rng)
    )
end

function LuxCore.initialstates(rng::AbstractRNG, l::LipschitzDense)
    return (;
        warmstart = Val(false),  
        training = Val(true), 
        u = randn(rng, Float32, l.out_dims), 
        v = randn(rng, Float32, l.in_dims),
        __scale_w = one(Float32),
        __cache_w = zeros(Float32, l.out_dims, l.in_dims),
    )
end

LuxCore.parameterlength(d::LipschitzDense) = d.out_dims * d.in_dims + d.out_dims + 1
LuxCore.statelength(d::LipschitzDense) = 4
LuxCore.outputsize(d::LipschitzDense, _, ::AbstractRNG) = (d.out_dims,)

function (l::LipschitzDense)(x::AbstractArray, ps, st::NamedTuple)
    if iswarmstart(st)
        st.__scale_w = one(eltype(ps.weight))
        st.__cache_w .= ps.weight
    elseif LuxOps.istraining(st)
        power_iteration!(st.u, st.v, ps.weight)
        largest_sv = dot(st.u, ps.weight, st.v)
        st.__scale_w = softplus(ps.ci[1]) / (largest_sv + l.eps)
        st.__cache_w .= ps.weight
        st.__cache_w .*= st.__scale_w
    end
    _x = Lux.Utils.make_abstract_matrix(x)
    y = Lux.Utils.matrix_to_array(
        fused_dense_bias_activation(l.activation, st.__cache_w, _x, ps.bias), x
    )
    return y, st
end

# Precompute the scaled weights (weight * __scale_w) once so inference can skip the per-call rescaling
prime_scaled_weights!(ps, st) = nothing
function prime_scaled_weights!(ps::NamedTuple, st::NamedTuple)
    if haskey(st, :__cache_w)
        st.__cache_w .= ps.weight
        st.__cache_w .*= st.__scale_w
    else
        for k in keys(st)
            haskey(ps, k) && prime_scaled_weights!(ps[k], st[k])
        end
    end
    return nothing
end
