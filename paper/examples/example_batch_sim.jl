## setup
using Printf, Statistics, ADTypes, CairoMakie
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

## run
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
sol = solve(prob, FBDF(; autodiff=AutoFiniteDiff()); callback=_dryout)

@info "solve finished" sol.retcode t_end_s=sol.t[end] n_reb_end=sol[sys.reb.nL][end]

## Visualization of results

t_min = sol.t ./ 60
z = range(0, H_col, N_stages)
T_sym = [sys.reb.T; [getproperty(sys, Symbol(:stage_, j)).T for j in 2:N_stages]]
t_snap = [0, 60, 90, 95, 100, 105].*60
cols_x = [:darkblue, :forestgreen, :orange]

fig = Figure(size=(900, 400), fontsize=14)
ax1 = Axis(fig[1,1]; limits=((0, t_min[end]), (0, 1)),
    xlabel=L"t \; / \; \mathrm{min}", ylabel=L"x_i \; / \; \mathrm{mol\,mol^{-1}}")
ax2 = Axis(fig[1,2]; limits=(nothing, (0, H_col)),
    xlabel=L"T \; / \; \mathrm{K}", ylabel=L"z \; / \; \mathrm{m}")
for i in eachindex(components)
    lines!(ax1, t_min, sol[sys.reb.x[i]]; color=cols_x[i])
    lines!(ax1, t_min, sol[sys.res.x[i]]; color=cols_x[i], linestyle=:dash)
end
Legend(fig[0,1],[[LineElement(color=c) for c in cols_x[1:length(components)]], [LineElement(color=:black), LineElement(color=:black, linestyle=:dash)]], [components, ["reboiler", "distillate"]], [nothing, nothing]; orientation=:horizontal, nbanks=2, framevisible=false)
for tk in t_snap
    scatterlines!(ax2, sol(tk; idxs=T_sym), z; color=tk, colormap=:darktest, colorrange=extrema(t_snap), label=@sprintf("%.0f min", tk/60))
end
Legend(fig[0,2], ax2; orientation=:horizontal, nbanks=2, framevisible=false)
save(joinpath(@__DIR__, "fig_batch.pdf"), fig)
fig
