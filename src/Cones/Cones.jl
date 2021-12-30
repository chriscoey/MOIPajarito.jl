# cut oracles for MathOptInterface cones

module Cones

import LinearAlgebra
import JuMP
const MOI = JuMP.MOI
const VR = JuMP.VariableRef
const AE = JuMP.AffExpr

abstract type NatExt end
struct Nat <: NatExt end
struct Ext <: NatExt end

function extender(extend::Bool, d::Int)
    if d <= 1 || !extend
        return Nat
    end
    return Ext
end

abstract type Cone end

include("secondordercone.jl")
include("exponentialcone.jl")
include("powercone.jl")
include("positivesemidefiniteconetriangle.jl")

# supported cones for outer approximation
const OACone = Union{
    MOI.SecondOrderCone,
    MOI.ExponentialCone,
    MOI.PowerCone{Float64},
    MOI.PositiveSemidefiniteConeTriangle,
}

setup_auxiliary(::Cone, ::JuMP.Model) = VR[]

extend_start(::Cone, ::Vector{Float64}) = Float64[]

num_ext_variables(::Cone) = 0

function dot_expr(
    z::AbstractVecOrMat{Float64},
    vars::AbstractVecOrMat{<:Union{VR, AE}},
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

function load_s(cache::Cone, ::Nothing)
    cache.s = JuMP.value.(cache.oa_s)
    return
end

function load_s(cache::Cone, cb::Any)
    return cache.s = JuMP.callback_value.(cb, cache.oa_s)
end

end
