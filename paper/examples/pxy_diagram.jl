using MLThermoProperties, Clapeyron, CairoMakie

comps = ["tetrachloromethane", "tert-butanol"]
model = multHANNA(comps; puremodel=GRAPPA)

T = 298.15
x₁ = range(0.0, 1.0, 51)
p, y₁ = zeros(51), zeros(51)
for (i, x) in enumerate(x₁)
    z = [x, 1-x]
    pᵢ, _, _, yᵢ = bubble_pressure(model, T, z)
    p[i], y₁[i] = pᵢ/1e5, yᵢ[1]
end

fig = Figure(size=(400, 280), fontsize=14)
ax = Axis(fig[1,1]; limits=((0,1), nothing),
    xlabel=L"x_1,\,y_1 \;/\; \mathrm{mol\,mol^{-1}}",
    ylabel=L"p \;/\; \mathrm{bar}")
lines!(ax, x₁, p; color=:blue, label="bubble")
lines!(ax, y₁, p; color=:red, label="dew")
axislegend(ax; position=:lt, framevisible=false)
save(joinpath(@__DIR__, "pxy_diagram.pdf"), fig)
