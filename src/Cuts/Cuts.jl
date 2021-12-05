# cut oracles for MathOptInterface cones

module Cuts

import LinearAlgebra
import JuMP
const MOI = JuMP.MOI
const AVS = MOI.AbstractVectorSet
const VR = JuMP.VariableRef
const CR = JuMP.ConstraintRef

import MOIPajarito: Optimizer

# initial fixed cuts (default to none)
add_init_cuts(::Optimizer, ::Vector{VR}, ::AVS) = 0

# strengthened subproblem dual cuts (default to no strengthening)
function add_subp_cuts(opt::Optimizer, z::Vector{Float64}, s_vars::Vector{VR}, ::AVS)
    expr = JuMP.@expression(opt.oa_model, JuMP.dot(z, s_vars))
    return add_cut(expr, opt)
end

# separation cuts (default to none)
function add_sep_cuts(::Optimizer, ::Vector{Float64}, ::Vector{VR}, ::AVS)
    return 0
end

# include("arrayutilities.jl")
include("secondordercone.jl")
# include("positivesemidefiniteconetriangle.jl")

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

end
