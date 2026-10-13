using Clapeyron
using Clapeyron: ActivityModel, EoSVectorParam,
                 init_puremodel, Rgas

struct MyGE{P} <: ActivityModel
    components::Vector{String}
    params::Vector{Float64}
    puremodel::EoSVectorParam{P}
    references::Vector{String}
end

function MyGE(comps; params, puremodel=PR)
    pure = init_puremodel(puremodel, comps,
                          String[], false)
    return MyGE(comps, params, pure, String[])
end

function Clapeyron.excess_gibbs_free_energy(
        model::MyGE, p, T, z)
    x = z ./ sum(z)
    A, B = model.params
    return sum(z)*Rgas(model)*T*x[1]*x[2]*(A+B/T)
end
