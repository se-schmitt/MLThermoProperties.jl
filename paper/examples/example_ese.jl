using MLThermoProperties, EntropyScaling
using Clapeyron

components = ["hexane", "benzene"]
vismodel = ESFramework(components, SAFTVRMie)
model = ESE(components; vismodel)

p, Tmin, Tmax = 1e5, 285., 340.
T = range(Tmin, Tmax, 50)
D₁₂ = inf_diffusion_coefficient.(model, p, T;
    solute="hexane", solvent="benzene")
D₂₁ = inf_diffusion_coefficient.(model, p, T;
    solute="benzene", solvent="hexane")

# Plotting ...
using CairoMakie, DataFrames, CSV
set_theme!(Axis=(xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1, xticksmirrored=true, yticksmirrored=true))

df = CSV.File(joinpath(@__DIR__, "Dinf_benzene_n-hexane_exp.csv")) |> DataFrame
what_D₁₂ = df.solvent .== "benzene" .&& df.solute .== "n-hexane"
what_D₂₁ = df.solvent .== "n-hexane" .&& df.solute .== "benzene"

fig = Figure(size=(400, 280), fontsize=14)
ax = Axis(fig[1,1]; 
    xlabel=rich(rich("T"; font=:italic), " / K"), 
    ylabel=rich(rich("D"; font=:italic), subsup(rich("ij"; font=:italic), "∞"), " / 10", superscript("−9"), " m", superscript("2"), " s", superscript("−1")), 
    limits=((Tmin,Tmax),nothing)
)
label₁₂ = rich(rich("D"; font=:italic), subsup("12", "∞"))
label₂₁ = rich(rich("D"; font=:italic), subsup("21", "∞"))
lines!(ax, T, D₁₂ .* 1e9; color=:blue, label=label₁₂)
scatter!(ax, df.var"T / K"[what_D₁₂], df.var"Dinf / 10⁻⁹ m² s⁻¹"[what_D₁₂]; marker=:circle, color=(:white,0.0), strokecolor=:blue, strokewidth=1, label=label₁₂)
lines!(ax, T, D₂₁ .* 1e9; color=:red, label=label₂₁)
scatter!(ax, df.var"T / K"[what_D₂₁], df.var"Dinf / 10⁻⁹ m² s⁻¹"[what_D₂₁]; marker=:circle, color=(:white,0.0), strokecolor=:red, strokewidth=1, label=label₂₁)
axislegend(ax; position=:lt, framevisible=false, merge=true,
    markerpoints=[Point2f(0.15, 0.5)], linepoints=[Point2f(0.45, 0.5), Point2f(1, 0.5)])
save(joinpath(@__DIR__, "fig_ese.pdf"), fig)
fig