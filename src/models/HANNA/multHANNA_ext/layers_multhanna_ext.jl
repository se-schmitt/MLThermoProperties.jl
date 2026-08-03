#using Lux, ConcreteStructs, Random, LinearAlgebra

@concrete struct multHANNA_extLux <: AbstractLuxContainerLayer{(:theta, :alpha, :phi)}
    theta
    alpha
    phi
    __cache_θs
end

Clapeyron.is_splittable(::multHANNA_extLux) = false

function (model::multHANNA_extLux)((T, x, embs), ps, st)
    N = length(x)
    
    θs = isnothing(model.__cache_θs) ?
        [first(model.theta(_emb, ps.theta, st.theta)) for _emb in embs] :
        model.__cache_θs

    cos_sim = ones(N, N)
    for i in 1:N
        for j in (i+1):N
            # Cosine similarity
            sim = cosine_similarity(θs[i], θs[j])
            cos_sim[i,j] = sim
            cos_sim[j,i] = sim
        end
    end
    
    x_adj = cos_sim * x
    
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
            correction = x[i] * x[j] * (1.0 - cos_sim[i, j])
            
            # adding correction 
            gE_total += gE_NN_ij * correction
        end
    end
    
    return gE_total, st
end