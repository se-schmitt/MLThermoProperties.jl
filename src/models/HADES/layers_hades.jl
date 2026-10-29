@concrete struct HADESLux <: AbstractLuxContainerLayer{(:enc, :alpha, :beta, :dec, :seb)}
    enc
    alpha
    beta
    dec
    seb
end

function HADESLux(; inp::Int=6, nodes::Int=32, out::Int=inp)
    enc = Dense(inp => nodes, silu)
    alpha = Dense(nodes => nodes, use_bias = false) 
    beta = Dense(nodes => nodes)    
    dec = Dense(nodes => out, softplus)
    seb = Chain(Dense(2*inp => 32, silu), Dense(32 => 16, silu), Dense(16 => 1, softplus))
    
    return HADESLux(enc, alpha, beta, dec, seb)
end

function (l::HADESLux)(inputs::Tuple, ps, st)
    x, z, smask = inputs

    M, st_enc = l.enc(x, ps.enc, st.enc)

    M_w = M .* z
    M_indiv, st_alpha = l.alpha(M_w, ps.alpha, st.alpha)
    M_g, st_beta = l.beta(sum(M_w, dims=2), ps.beta, st.beta)
    M = silu.(M_indiv .+ M_g)
    M, st_dec = l.dec(M, ps.dec, st.dec)
    mixed_solvent = sum(x .* z .* M, dims=2) 
    solute = x[:, 1:1, :] 
    
    seb_input = vcat(solute, mixed_solvent) 
    cf, st_seb = l.seb(seb_input, ps.seb, st.seb)
    
    st_new = (enc=st_enc, alpha=st_alpha, beta=st_beta, dec=st_dec, seb=st_seb)
    
    return cf, st_new
end