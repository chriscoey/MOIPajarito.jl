# K* cut utilities
# TODO improve using functions in MOI Utilities/functions.jl

# initial fixed cuts
function add_init_cuts(opt::Optimizer)
    for (ci, s_vars, cone) in opt.zip_cones
        cuts = Cuts.get_init_cuts(cone)
        for cut in cuts
            _add_constraint(cut, s_vars, opt.oa_opt, nothing)
        end
        opt.num_cuts += length(cuts)
    end
    return nothing
end

# subproblem dual cuts
function add_subp_cuts(opt::Optimizer, cb = nothing)
    num_cuts_before = opt.num_cuts
    for (ci, s_vars, cone) in opt.zip_cones
        # dual = JuMP.dual(ci)
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
        @show dual

        cuts = Cuts.get_subp_cuts(dual, cone, opt.tol_feas)
        for cut in cuts
            @show cut
            _add_constraint(cut, s_vars, opt.oa_opt, cb)
        end
        opt.num_cuts += length(cuts)
    end
    return (opt.num_cuts > num_cuts_before)
end

# separation cuts
function add_sep_cuts(opt::Optimizer, cb = nothing)
    num_cuts_before = opt.num_cuts
    for (ci, s_vars, cone) in opt.zip_cones
        prim = _get_value(s_vars)
        @assert !any(isnan, prim)
        cuts = Cuts.get_sep_cuts(prim, cone, opt.tol_feas)
        for cut in cuts
            _add_constraint(cut, s_vars, opt.oa_opt, cb)
        end
        opt.num_cuts += length(cuts)
    end
    return (opt.num_cuts > num_cuts_before)
end

# helpers

function _get_value(
    s_vars::Vector{VI},
    oa_opt::MOI.ModelLike,
    ::Nothing,
    )
    return JuMP.value.(s_vars)
end

function _get_value(
    s_vars::Vector{VI},
    oa_opt::MOI.ModelLike,
    cb,
    )
    return JuMP.callback_value.(cb, s_vars)
end

function _add_constraint(
    cut::Vector{Float64},
    s_vars::Vector{VI},
    oa_opt::MOI.ModelLike,
    ::Nothing,
)
    return JuMP.@constraint(oa_opt, JuMP.dot(cut, s_vars) >= 0)
end

function _add_constraint(
    cut::Vector{Float64},
    s_vars::Vector{VI},
    oa_opt::MOI.ModelLike,
    cb,
)
    con = JuMP.@build_constraint(JuMP.dot(cut, s_vars) >= 0)
    return MOI.submit(oa_opt, MOI.LazyConstraint(cb), con)
end
