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
const VR = JuMP.VariableRef
const CR = JuMP.ConstraintRef
const AE = JuMP.AffExpr

include("Cones/Cones.jl")
include("optimizer.jl")
include("algorithms.jl")
include("models.jl")
include("cuts.jl")
include("JuMP_tools.jl")
include("MOI_wrapper.jl")
include("MOI_copy_to.jl")

end
