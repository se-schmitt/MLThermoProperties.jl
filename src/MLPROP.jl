module MLPROP

using Clapeyron, Lux, ConcreteStructs
using MLPROP_EOS

import RDKitMinimalLib
import DataAndTrainingUtils: scale, unscale, RobustScaler, FixedZeroRobustScaler

const CL = Clapeyron
const RDK = RDKitMinimalLib

const R̄32 = Float32(CL.R̄)

# Models
include("HANNA/hanna.jl")
include("EOS/romeos.jl")
include("EOS/romeos_id.jl")

end
