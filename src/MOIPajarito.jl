# a solver for mixed-integer conic problems using outer approximation and conic duality

module MOIPajarito

include("Cuts/Cuts.jl")

import Printf
import LinearAlgebra

import JuMP
const MOI = JuMP.MOI
# TODO delete unused
const MOIU = MOI.Utilities
const VI = MOI.VariableIndex
const CI = MOI.ConstraintIndex
const SAT = MOI.ScalarAffineTerm{Float64}
const SAF = MOI.ScalarAffineFunction{Float64}
const VV = MOI.VectorOfVariables
const VAT = MOI.VectorAffineTerm{Float64}
const VAF = MOI.VectorAffineFunction{Float64}

include("optimize.jl")
include("cut_utilities.jl")
include("MOI_wrapper.jl")

end
