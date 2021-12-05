# K* cut utilities
# TODO improve using functions in MOI Utilities/functions.jl

# initial fixed cuts
function add_init_cuts(opt::Optimizer)
    for (ci, s_vars, cone) in opt.oa_cones
        cuts = Cuts.get_init_cuts(cone)
        for cut in cuts
            JuMP.@constraint(opt.oa_model, JuMP.dot(cut, s_vars) >= 0)
        end
        opt.num_cuts += length(cuts)
    end
    return nothing
end

# subproblem dual cuts
function add_subp_cuts(opt::Optimizer, cb = nothing)
    num_cuts_before = opt.num_cuts
    for (ci, s_vars, cone) in opt.oa_cones
        dual = JuMP.dual(ci)
        @assert !any(isnan, dual)

        # TODO tune
        norm_dual = LinearAlgebra.norm(dual, Inf)
        if norm_dual < opt.tol_feas
            # discard duals with small norm_dual
            continue
        else
            # rescale dual to have norm 1
            LinearAlgebra.lmul!(inv(norm_dual), dual)
        end

        cuts = Cuts.get_subp_cuts(dual, cone, opt.tol_feas)
        for cut in cuts
            opt.num_cuts += _add_constraint(cut, s_vars, opt, cb)
        end
    end
    return (opt.num_cuts > num_cuts_before)
end

# separation cuts
function add_sep_cuts(opt::Optimizer, cb = nothing)
    num_cuts_before = opt.num_cuts
    for (ci, s_vars, cone) in opt.oa_cones
        prim = _get_value(s_vars, cb)
        @assert !any(isnan, prim)

        cuts = Cuts.get_sep_cuts(prim, cone, opt.tol_feas)
        for cut in cuts
            opt.num_cuts += _add_constraint(cut, s_vars, opt, cb)
        end
    end
    return (opt.num_cuts > num_cuts_before)
end

# helpers

function _add_constraint(
    cut::Vector{Float64},
    s_vars::Vector{JuMP.VariableRef},
    opt::Optimizer,
    ::Nothing,
)
    JuMP.@constraint(opt.oa_model, JuMP.dot(cut, s_vars) >= 0)
    return 1
end

function _add_constraint(
    cut::Vector{Float64},
    s_vars::Vector{JuMP.VariableRef},
    opt::Optimizer,
    cb,
)
    # only add cut if violated (per JuMP documentation)
    expr = JuMP.dot(cut, s_vars)
    if JuMP.callback_value(cb, expr) < opt.tol_feas
        con = JuMP.@build_constraint(expr >= 0)
        MOI.submit(opt.oa_model, MOI.LazyConstraint(cb), con)
        return 1
    end
    return 0
end
