using MLThermoProperties, Clapeyron, CairoMakie

comps = ["tetrachloromethane", "tert-butanol"]
model = multHANNA(comps; puremodel=GRAPPA)

T = 298.15
x₁ = range(0.0, 1.0, 51)
p, y₁ = zeros(51), zeros(51)
lnγ₁, lnγ₂ = zeros(51), zeros(51)
for (i, x) in enumerate(x₁)
    z = [x, 1-x]
    pᵢ, _, _, yᵢ = bubble_pressure(model, T, z)
    p[i], y₁[i] = pᵢ/1e5, yᵢ[1]
    γᵢ = activity_coefficient(model, 1e5, T, z)
    lnγ₁[i], lnγ₂[i] = log.(γᵢ)
end

# Plotting ...
fig = Figure(size=(400, 500), fontsize=14)
ax1 = Axis(fig[1,1]; limits=((0,1), nothing),
    xticklabelsvisible=false,
    ylabel=L"p / \mathrm{bar}")
ax2 = Axis(fig[2,1]; limits=((0,1), nothing),
    xlabel=L"x_1,y_1 / \mathrm{mol\,mol^{-1}}",
    ylabel=L"\ln \gamma_i")
lines!(ax1, x₁, p; color=:blue, label="bubble")
lines!(ax1, y₁, p; color=:red, label="dew")
axislegend(ax1; position=:lt, framevisible=false)
lines!(ax2, x₁, lnγ₁; color=:navy, label="γ₁")
lines!(ax2, x₁, lnγ₂; color=:green, label="γ₂")
axislegend(ax2; position=:lt, framevisible=false)
save(joinpath(@__DIR__, "fig_hanna.pdf"), fig)
