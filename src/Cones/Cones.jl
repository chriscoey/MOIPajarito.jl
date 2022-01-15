# cut oracles for MathOptInterface cones

module Cones

import LinearAlgebra
import JuMP
const MOI = JuMP.MOI
const VR = JuMP.VariableRef
const AE = JuMP.AffExpr

import MOIPajarito: Cache, Optimizer

abstract type NatExt end
struct Nat <: NatExt end
struct Ext <: NatExt end
nat_or_ext(extend::Bool, d::Int) = ((d <= 1 || !extend) ? Nat : Ext)

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

setup_auxiliary(::Cache, ::Optimizer) = VR[]

extend_start(::Cache, ::Vector{Float64}) = Float64[]

num_ext_variables(::Cache) = 0

function dot_expr(
    z::AbstractVecOrMat{Float64},
    vars::AbstractVecOrMat{<:Union{VR, AE}},
    opt::Optimizer,
)
    return JuMP.@expression(opt.oa_model, JuMP.dot(z, vars))
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

get_oa_s(cache::Cache) = cache.oa_s

end
