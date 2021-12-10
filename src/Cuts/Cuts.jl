# cut oracles for MathOptInterface cones

module Cuts

import LinearAlgebra
import JuMP
const MOI = JuMP.MOI
const AVS = MOI.AbstractVectorSet
const VR = JuMP.VariableRef
const CR = JuMP.ConstraintRef

import MOIPajarito: Optimizer

include("secondordercone.jl")
include("exponentialcone.jl")
include("powercone.jl")
include("positivesemidefiniteconetriangle.jl")

# initial fixed cuts (default to none)
# add_init_cuts(::Optimizer, ::Vector{VR}, ::AVS) = 0

# strengthened subproblem dual cuts (default to no strengthening)
# function add_subp_cuts(opt::Optimizer, z::Vector{Float64}, s_vars::Vector{VR}, ::AVS)
#     # TODO MOI.Utilities.set_dot
#     expr = JuMP.@expression(opt.oa_model, JuMP.dot(z, s_vars))
#     return add_cut(expr, opt)
# end

# separation cuts (default to none)
# add_sep_cuts(::Optimizer, ::Vector{Float64}, ::Vector{VR}, ::AVS) = 0

function dot_expr(
    z::LinearAlgebra.AbstractVecOrMat{Float64},
    s_vars::AbstractVecOrMat{VR},
    opt::Optimizer,
)
    return JuMP.@expression(opt.oa_model, JuMP.dot(z, s_vars))
end

function add_cut(expr::JuMP.AffExpr, opt::Optimizer)
    return _add_cut(expr, opt.oa_model, opt.tol_feas, opt.lazy_cb)
end

function _add_cut(expr::JuMP.AffExpr, model::JuMP.Model, ::Float64, ::Nothing)
    JuMP.@constraint(model, expr >= 0)
    return 1
end

function _add_cut(expr::JuMP.AffExpr, model::JuMP.Model, tol_feas::Float64, cb)
    # only add cut if violated (per JuMP documentation)
    if JuMP.callback_value(cb, expr) < -tol_feas
        con = JuMP.@build_constraint(expr >= 0)
        MOI.submit(model, MOI.LazyConstraint(cb), con)
        return 1
    end
    return 0
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

end
