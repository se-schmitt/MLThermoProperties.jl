module EntropyScalingExt

using MLPROP
using EntropyScaling

const ES = EntropyScaling

function MLPROP.ESE(components, es_model::ES.AbstractEntropyScalingModel; p=1e5, kwargs...)
    wrapper = ESModelWrapper(es_model, p)
    return ESE(components, wrapper; kwargs...)
end

struct ESModelWrapper{M,T}
    model::M
    p::T
end
function (wrapper::ESModelWrapper{M,_T})(T) where {M,_T}
    return viscosity(wrapper.model, wrapper.p, T)
end

end