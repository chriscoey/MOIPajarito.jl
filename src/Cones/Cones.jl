# cut oracles for MathOptInterface cones

module Cones

import LinearAlgebra
import JuMP
const MOI = JuMP.MOI
const AVS = MOI.AbstractVectorSet
const VR = JuMP.VariableRef
const CR = JuMP.ConstraintRef

abstract type Extender end
struct Unextended <: Extender end
struct Extended <: Extender end

extender(extend::Bool) = (extend ? Extended : Unextended)

abstract type ConeCache end

include("secondordercone.jl")
# include("exponentialcone.jl")
# include("powercone.jl")
# include("positivesemidefiniteconetriangle.jl")

# supported cones for outer approximation
const OACone = Union{
    MOI.SecondOrderCone,
    # MOI.ExponentialCone,
    # MOI.PowerCone{Float64},
    # MOI.PositiveSemidefiniteConeTriangle,
}

setup_auxiliary(::ConeCache, ::JuMP.Model) = nothing

function dot_expr(
    z::AbstractVecOrMat{Float64},
    vars::AbstractVecOrMat{VR},
    oa_model::JuMP.Model,
)
    return JuMP.@expression(oa_model, JuMP.dot(z, vars))
end

function clean_array!(z::AbstractArray)
    # avoid poorly conditioned cuts and near-zero values
    z_norm = LinearAlgebra.norm(z, Inf)
    min_abs = max(1e-12, 1e-15 * z_norm) # TODO tune/option
    for (i, z_i) in enumerate(z)
        if abs(z_i) < min_abs
            z[i] = 0
        end
    end
    return iszero(z)
end

function load_s(cache::ConeCache, ::Nothing)
    cache.s = JuMP.value.(cache.s_oa)
    return
end

function load_s(cache::ConeCache, cb::Any)
    cache.s = JuMP.callback_value.(cb, cache.s_oa)
end

end