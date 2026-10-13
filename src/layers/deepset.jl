# simulates "EquivLinear" from Python (JW)
struct EquivLinear{L1, L2} <: AbstractLuxContainerLayer{(:alpha_layer, :beta_layer)}
    alpha_layer::L1 # X * alpha
    beta_layer::L2  # sum(X) * beta + bias
end

function EquivLinear(in_features::Int, out_features::Int)

    alpha_layer = Dense(in_features => out_features, use_bias=false)
    beta_layer = Dense(in_features => out_features)
    
    return EquivLinear(alpha_layer, beta_layer)
end

# forward
function (l::EquivLinear)(x::AbstractArray, mole_fractions::AbstractArray, smask::AbstractArray, ps, st)

    x_weighted = x .* mole_fractions 
    
    global_sum = sum(x_weighted, dims=2)
    
    global_context, st_beta = l.beta_layer(global_sum, ps.beta_layer, st.beta_layer)
    
    indiv_transformed, st_alpha = l.alpha_layer(x_weighted, ps.alpha_layer, st.alpha_layer)
    
    out = indiv_transformed .+ global_context
    
    out = out .* smask
    
    st_new = (alpha_layer=st_alpha, beta_layer=st_beta)
    return out, st_new
end