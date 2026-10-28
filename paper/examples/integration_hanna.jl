Clapeyron.excess_gibbs_free_energy(
    model::ogHANNA, p, T, z
) = begin
    x = z ./ sum(z)
    
    params = model.params
    Ts = scale(params.scaler_T, T)
    gE = params.nn((Ts,x,params.θs.values))

    return gE * Rgas(model) * T * sum(z)
end
