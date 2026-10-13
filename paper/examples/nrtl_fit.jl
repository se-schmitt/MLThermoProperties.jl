using MLThermoProperties, Clapeyron
using Optim, CairoMakie

comps = ["tetrachloromethane", "tert-butanol"]
T, p = 298.15, 1e5
x₁ = range(0.01, 0.99, 21)
γ(m, x) = activity_coefficient(m, p, T, [x, 1-x])

hanna = multHANNA(comps)
lnγ_pred = [log.(γ(hanna, x)) for x in x₁]

nrtl = NRTL(comps)
function loss(θ)
    nrtl.params.a.values .= [0.0 θ[1]; θ[2] 0.0]
    nrtl.params.c.values .= [0.0 θ[3]; θ[3] 0.0]
    err = sum(sum(abs2, lnγ .- log.(γ(nrtl, x)))
              for (x, lnγ) in zip(x₁, lnγ_pred))
    return err / (2*length(x₁))
end

res = optimize(loss, [1.0, 1.0, 0.3], NelderMead())
θ = Optim.minimizer(res)
loss(θ)

lnγ_nrtl = [log.(γ(nrtl, x)) for x in x₁]

fig = Figure(size=(400, 280), fontsize=14)
ax = Axis(fig[1,1]; limits=((0,1), nothing),
    xlabel=L"x_1 \;/\; \mathrm{mol\,mol^{-1}}",
    ylabel=L"\ln \gamma_i")
for i in 1:2
    c = (:blue, :red)[i]
    scatter!(ax, x₁, getindex.(lnγ_pred, i); color=c,
        label=i == 1 ? "HANNA" : nothing)
    lines!(ax, x₁, getindex.(lnγ_nrtl, i); color=c,
        linestyle=:dash, label=i == 1 ? "NRTL fit" : nothing)
end
axislegend(ax; position=:ct, framevisible=false)
save(joinpath(@__DIR__, "nrtl_fit.pdf"), fig)
