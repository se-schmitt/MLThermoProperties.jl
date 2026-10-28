EntropyScaling._inf_diffusion_coefficient(
    model::ESE, p, T, (idx_i,idx_j); phase=:liquid
) = begin
    f = 0.64
    ϱ_ref = 1050.
    M_i = model.params.Mw.values[idx_i]*1e-3
    pure_vismodel = model.pure_vismodels[idx_j]
    η_j = viscosity(pure_vismodel, p, T; phase)

    r_i = cbrt(f * 3 * M_i / (4π * ϱ_ref * NA))
    Dij_SE = (kB*T)/(6π*η_j*r_i)
    return Dij_SE * model.params.b_ij[idx_i,idx_j]
end
