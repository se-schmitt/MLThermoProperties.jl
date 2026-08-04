abstract type AbstractMultHANNALux{layers} <: AbstractLuxContainerLayer{layers} end

@concrete struct multHANNALux <: AbstractMultHANNALux{(:theta, :alpha, :phi)}
    theta
    alpha
    phi
    __cache_θs
    gamma
end

Clapeyron.is_splittable(::multHANNALux) = false

function (model::AbstractMultHANNALux)((T, x, embs), ps, st)
    N = length(x)
    
    θs = isnothing(model.__cache_θs) ?
        [first(model.theta(_emb, ps.theta, st.theta)) for _emb in embs] :
        model.__cache_θs

    similarity = ones(N,N)
    calc_similarity!(similarity, model, θs)
    
    x_adj = similarity * x
    
    gE_total = zero(Base.promote_eltype(T,x))
    
    for i in 1:N
        for j in (i+1):N
            # Muggianu
            X_i_ij = (1.0 + x_adj[i] - x_adj[j]) / 2.0
            X_j_ij = (1.0 + x_adj[j] - x_adj[i]) / 2.0
            
            # Alpha input, adding pair interaction and temperature
            c_i = vcat(θs[i], X_i_ij, T)
            c_j = vcat(θs[j], X_j_ij, T)
            
            α_i = first(model.alpha(c_i, ps.alpha, st.alpha))
            α_j = first(model.alpha(c_j, ps.alpha, st.alpha))
            α_ij = α_i .+ α_j 
            
            # phi
            gE_NN_ij = first(model.phi(α_ij, ps.phi, st.phi))[1]
            
            # check simalarity
            correction = x[i] * x[j] * (1.0 - similarity[i, j])
            
            # adding correction 
            gE_total += gE_NN_ij * correction
        end
    end
    
    return gE_total, st
end

function calc_similarity!(similarity, model::multHANNALux, θs)
    N = length(θs)
    for i in 1:N, j in (i+1):N
        similarity[i,j] = exp(-model.gamma * sum(abs2, θs[i] .- θs[j]))
        similarity[j,i] = similarity[i,j]
    end
end