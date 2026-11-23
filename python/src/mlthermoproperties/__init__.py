from pyclapeyron import jl
from pyclapeyron.utils import create_wrapper

jl.seval("using MLThermoProperties, EntropyScaling, CoolProp")


def _export(mod):
    for name in map(str, jl.names(mod)):
        if name.startswith("@") or not jl.isdefined(mod, jl.Symbol(name)):
            continue
        obj = getattr(mod, name)
        if jl.isa(obj, jl.Function):
            obj = create_wrapper(obj)
            name = obj.__name__
        globals()[name] = obj


_export(jl.EntropyScaling)
_export(jl.MLThermoProperties)
