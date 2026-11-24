using Printf, Statistics, ADTypes
using ModelingToolkit, SciMLBase, OrdinaryDiffEq
using Clapeyron, MLThermoProperties
using BatchDistillation

# input
n_dry  = 1.     # mol, mininmum reboiler holdup
x_pure = 1e-4   # mol mol⁻¹, purity criteria

# system
components = [
    "methyl benzoate", 
    "ethyl benzoate", 
    "dimethylphthalate"
]

# thermodynamic model 
model = CompositeModel(components; 
    liquid=COGNAC, 
    fluid=CompositeModel(components; 
        liquid=DIPPR105Liquid, 
        saturation=GRAPPA, 
        gas=TsonopoulosVirial
    )
)

# plant-specific parameters
N_stages = 12
H_col    = 1.5  # m, height of column

# experiment-specific parameters
Q    = 300.     # W, reboiler heat duty
PS   = 10000.0  # Pa, pressure in column
ϵ    = 0.50     # efflux ratio
x_charge  = [0.5, 0.3, 0.2] # reboiler charge
nB   = 3.3      # mol, buffer holdup
n_charge = 20   # mol, reboiler charge

# create column
@named col = Column(; fluid=model, N_stages)
sys = mtkcompile(col)

# parameter map
params = Dict(sys.reb.p => PS, sys.reb.Q => Q,
	sys.cond.p => PS, sys.cond.nB => nB, 
    sys.cond.ϵ => ϵ, 
	(getproperty(sys, Symbol(:stage_, j)).p => PS 
        for j in 2:N_stages)...,
)

# initialization
init = initialize_total_reflux(sys; 
    fluid=model, N_stages, 
    p=PS, Q, x_charge, n_charge
)

params0 = merge(init.u0, params)
prob = ODEProblem(sys, params0, (0.0, 7200.); 
    guesses = init.guesses
)

# callbacks/events
reb_nL = ModelingToolkit.getu(sys, sys.reb.nL)
reb_x = ModelingToolkit.getu(sys, 
    collect(sys.reb.x)
)
_dryout = DiscreteCallback(
    (u,t,i) -> reb_nL(i) < n_dry || 
        maximum(reb_x(i)) > 0.9999,
    terminate!; save_positions = (true, false)
)

# solve
solver = FBDF(; autodiff=AutoFiniteDiff())
sol = solve(prob, solver; 
    callback=_dryout
)

# Plotting...
using CairoMakie
set_theme!(Axis=(xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1, xticksmirrored=true, yticksmirrored=true))

t_min = sol.t ./ 60
z = range(0, H_col, N_stages)
T_sym = [sys.reb.T; [getproperty(sys, Symbol(:stage_, j)).T for j in 2:N_stages]]
t_snap = [0, 60, 90, 95, 100, 105].*60
cols_x = [:grey65, :grey45, :grey20]

fig = Figure(size=(900, 400), fontsize=14)
ax1 = Axis(fig[1,1]; limits=((0, t_min[end]), (0, 1)),
    xlabel=rich(rich("t"; font=:italic), " / min"), ylabel=rich(rich("x"; font=:italic), subscript("i"; font=:italic), " / mol mol", superscript("−1")))
ax2 = Axis(fig[1,2]; limits=(nothing, (0, H_col)),
    xlabel=rich(rich("T"; font=:italic), " / K"), ylabel=rich(rich("z"; font=:italic), " / m"))
for i in eachindex(components)
    lines!(ax1, t_min, sol[sys.reb.x[i]]; color=cols_x[i])
    lines!(ax1, t_min, sol[sys.res.x[i]]; color=cols_x[i], linestyle=:dash)
end
snap_pos = lift(ax1.scene.viewport, ax1.finallimits) do vp, lims
    Point2f.(vp.origin[1] .+ vp.widths[1] .* (t_snap./60 .- lims.origin[1]) ./ lims.widths[1], vp.origin[2] + vp.widths[2])
end
scatter!(fig.scene, snap_pos; marker='↓', color=t_snap, colormap=:darktest, colorrange=extrema(t_snap), markersize=20, marker_offset=(0, 9))
Legend(fig[0,1],[[LineElement(color=c) for c in cols_x[1:length(components)]], [LineElement(color=:black), LineElement(color=:black, linestyle=:dash)]], [replace.(components, "dimethylphthalate" => "dimethyl phthalate"), ["reboiler", "distillate"]], [nothing, nothing]; orientation=:horizontal, nbanks=2, framevisible=false)
for tk in t_snap
    scatterlines!(ax2, sol(tk; idxs=T_sym), z; color=tk, colormap=:darktest, colorrange=extrema(t_snap), label=@sprintf("%.0f min", tk/60))
end
Legend(fig[0,2], ax2; orientation=:horizontal, nbanks=2, framevisible=false)
save(joinpath(@__DIR__, "fig_batch.pdf"), fig)
fig
