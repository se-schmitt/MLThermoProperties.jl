module MLPROP

using Clapeyron, Lux, ConcreteStructs,JLD2
#using ROMEOSdev

using RDKitMinimalLib
#import DataAndTrainingUtils: scale, unscale, RobustScaler, FixedZeroRobustScaler

const CL = Clapeyron
const RDK = RDKitMinimalLib

const R̄32 = Float32(CL.R̄)

const DB_PATH = normpath(Base.pkgdir(MLPROP),"database")

# Models
include("HANNA/hanna.jl")
include("EOS/romeos.jl")
include("EOS/romeos_id.jl")
include("models/SEB.jl")

end
