include("utils.jl")

include("ese.jl")
include("HADES/hades.jl")
include("GRAPPA/grappa.jl")
include("HANNA/hanna.jl")
include("COGNAC/cognac.jl")

# Show method
const MODELS = Union{GRAPPA, ogHANNA, multHANNA, ESE, COGNACexp, COGNACimp}

function Base.show(io::IO, ::MIME"text/plain", model::MODELS)
    print(io, nameof(typeof(model)))
    length(model) == 1 && println(io, " with 1 component:")
    length(model) > 1 && println(io, " with ", length(model), " components:")
    CL.show_pairs(io,CL.component_list(model))
    
    CL.show_info(io,model)
    CL.show_params(io,model)
    CL.show_reference_state(io,model)
    CL.may_show_references(io,model)
end
CL.show_reference_state(io,::ESE) = nothing
