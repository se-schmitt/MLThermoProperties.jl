using MLThermoProperties, EntropyScaling
using CoolProp, CairoMakie

model = ESE(["ethanol", "n-decane"])

p = 1e5
T = range(280.0, 340.0, 40)
D₁₂ = inf_diffusion_coefficient.(model, p, T;
    solute="ethanol", solvent="n-decane")
D₂₁ = inf_diffusion_coefficient.(model, p, T;
    solute="n-decane", solvent="ethanol")

# Plotting ...
fig = Figure(size=(400, 280), fontsize=14)
ylabel = L"D^\infty / 10^{-9}\mathrm{m^2 s^{-1}}"
ax = Axis(fig[1,1]; xlabel=L"T / \mathrm{K}", 
    ylabel)
lines!(ax, T, D₁₂ .* 1e9; color=:blue,
    label="ethanol in n-decane")
lines!(ax, T, D₂₁ .* 1e9; color=:red,
    label="n-decane in ethanol")
axislegend(ax; position=:lt, framevisible=false)
save(joinpath(@__DIR__, "fig_ese.pdf"), fig)
