@concrete struct HADESLux <: AbstractLuxContainerLayer{(:enc, :equ, :dec, :seb)}
    enc
    equ
    dec
    seb
end

function HADESLux(; inp::Int=6, nodes::Int=32, out::Int=inp)
    enc = Dense(inp => nodes, silu)
    equ = EquivLinear(nodes, nodes)
    dec = Dense(nodes => out, softplus)
    seb = Chain(Dense(2*inp => 32, silu), Dense(32 => 16, silu), Dense(16 => 1, softplus))
    
    return HADESLux(enc, equ, dec, seb)
end

function (l::HADESLux)(inputs::Tuple, ps, st)
    x, z, smask = inputs

    M, st_enc = l.enc(x, ps.enc, st.enc)
    M, st_equ = l.equ(M, z, smask, ps.equ, st.equ)
    M = silu.(M)
    M, st_dec = l.dec(M, ps.dec, st.dec)
    mixed_solvent = sum(x .* z .* M, dims=2) 
    solute = x[:, 1:1, :] 
    
    seb_input = vcat(solute, mixed_solvent) 
    cf, st_seb = l.seb(seb_input, ps.seb, st.seb)
    
    st_new = (enc=st_enc, equ=st_equ, dec=st_dec, seb=st_seb)
    
    return cf, st_new
end