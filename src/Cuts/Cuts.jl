# cut oracles for MathOptInterface cones
module Cuts

using LinearAlgebra
# import LinearAlgebra.copytri!

import MathOptInterface
const MOI = MathOptInterface
# const MOIU = MathOptInterface.Utilities

# initial fixed cuts (default to none)
get_init_cuts(::MOI.AbstractVectorSet) = Vector{Float64}[]

# strengthened subproblem dual cuts (default to identity)
get_subp_cuts(dual::Vector{Float64}, ::MOI.AbstractVectorSet, ::Float64) = [dual]

# separation cuts (default to none)
get_sep_cuts(::Vector{Float64}, ::MOI.AbstractVectorSet, ::Float64) = Vector{Float64}[]

# include("arrayutilities.jl")
include("secondordercone.jl")
# include("positivesemidefiniteconetriangle.jl")

end
