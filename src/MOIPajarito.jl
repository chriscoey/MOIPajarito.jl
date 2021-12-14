# a solver for mixed-integer conic problems using outer approximation and conic duality

module MOIPajarito

import Printf
import LinearAlgebra
import SparseArrays

import JuMP
const MOI = JuMP.MOI
const VI = MOI.VariableIndex
const SAF = MOI.ScalarAffineFunction{Float64}
const VV = MOI.VectorOfVariables
const VAF = MOI.VectorAffineFunction{Float64}
const AVS = MOI.AbstractVectorSet
const VR = JuMP.VariableRef
const CR = JuMP.ConstraintRef

include("Cones/Cones.jl")
include("optimize.jl")
include("MOI_wrapper.jl")

end
