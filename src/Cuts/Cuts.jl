# cut oracles for MathOptInterface cones
module Cuts

using LinearAlgebra
# import LinearAlgebra.copytri!

import MathOptInterface
const MOI = MathOptInterface
# const MOIU = MathOptInterface.Utilities

# include("arrayutilities.jl")
include("secondordercone.jl")
# include("positivesemidefiniteconetriangle.jl")

# fallbacks
get_init_cuts(::MOI.AbstractVectorSet) = Vector{Float64}[]
get_sep_cuts(::Vector{Float64}, ::MOI.AbstractVectorSet, ::Float64) = Vector{Float64}[]

end
