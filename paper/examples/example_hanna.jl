using MLThermoProperties, Clapeyron

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
using CairoMakie, DataFrames, CSV
set_theme!(Axis=(xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1, xticksmirrored=true, yticksmirrored=true))

df = CSV.File(joinpath(@__DIR__, "VLE_tetrachloromethane_tert-butanol_exp.csv")) |> DataFrame

fig = Figure(size=(400, 500), fontsize=14)
ax1 = Axis(fig[1,1]; limits=((0,1), nothing),
    xticklabelsvisible=false,
    ylabel=rich(rich("p"; font=:italic), " / bar"))
ax2 = Axis(fig[2,1]; limits=((0,1), nothing),
    xlabel=rich(rich("x"; font=:italic), subscript("1"), ", ", rich("y"; font=:italic), subscript("1"), " / mol mol", superscript("−1")),
    ylabel=rich("ln ", rich("γ"; font=:italic), subscript("i"; font=:italic)))
lines!(ax1, x₁, p; color=:black)
scatter!(ax1, df.x1, df.var"p / Pa" ./ 1e5; marker=:rect, color=(:white,0.0), strokecolor=:black, strokewidth=1, label="bubble")
lines!(ax1, y₁, p; color=:black)
scatter!(ax1, df.y1, df.var"p / Pa" ./ 1e5; marker=:circle, color=(:white,0.0), strokecolor=:black, strokewidth=1, label="dew")
axislegend(ax1; position=:lt, framevisible=false, merge=true,
    markerpoints=[Point2f(0.15, 0.5)], linepoints=[Point2f(0.45, 0.5), Point2f(1, 0.5)])
lines!(ax2, x₁, lnγ₁; color=:blue, label=rich(rich("γ"; font=:italic), subscript("1")))
lines!(ax2, x₁, lnγ₂; color=:red, label=rich(rich("γ"; font=:italic), subscript("2")))
axislegend(ax2; position=:lt, framevisible=false)
save(joinpath(@__DIR__, "fig_hanna.pdf"), fig)
fig