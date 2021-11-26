# K* cut utilities

# initial fixed cuts
function add_init_cuts(opt::Optimizer)
    for (ci, func, set) in opt.approx_cons
        cuts = Cuts.get_init_cuts(set)
        for cut in cuts
            _add_constraint(cut, func, opt.oa_opt, nothing)
        end
        opt.num_cuts += length(cuts)
    end
    return nothing
end

# subproblem dual cuts
function add_subp_cuts(opt::Optimizer, cb = nothing)
    num_cuts_before = opt.num_cuts
    for (ci, func, set) in opt.approx_cons
        dual = MOI.get(opt.conic_opt, MOI.ConstraintDual(), ci)
        @assert !any(isnan, dual)
        cuts = Cuts.get_subp_cuts(dual, set, opt.tol_feas)
        for cut in cuts
            _add_constraint(cut, func, opt.oa_opt, cb)
        end
        opt.num_cuts += length(cuts)
    end
    return (opt.num_cuts > num_cuts_before)
end

# separation cuts
function add_sep_cuts(opt::Optimizer, cb = nothing)
    num_cuts_before = opt.num_cuts
    for (ci, func, set) in opt.approx_cons
        # prim = MOIU.eval_variables(vi -> _get_val(vi, opt.oa_opt, cb), func)
        prim = MOI.get(opt.conic_opt, MOI.ConstraintPrimal(), ci)
        @assert !any(isnan, prim)
        cuts = Cuts.get_sep_cuts(prim, set, opt.tol_feas)
        for cut in cuts
            _add_constraint(cut, func, opt.oa_opt, cb)
        end
        opt.num_cuts += length(cuts)
    end
    return (opt.num_cuts > num_cuts_before)
end

# helpers

# function _get_val(vi::VI, oa_opt::MOI.ModelLike, ::Nothing)
#     return MOI.get(oa_opt, MOI.VariablePrimal(), vi)
# end

# function _get_val(vi::VI, oa_opt::MOI.ModelLike, cb)
#     return MOI.get(oa_opt, MOI.CallbackVariablePrimal(cb), vi)
# end

function _cut_expr(cut::Vector{Float64}, func::VV)
    return MOI.ScalarAffineFunction(SAT.(cut, func.variables), 0.0)
end

function _cut_expr(cut::Vector{Float64}, func::VAF)
    sats = [_cut_term(cut, t) for t in func.terms]
    return MOI.ScalarAffineFunction(sats, LinearAlgebra.dot(cut, func.constants))
end

function _cut_term(cut::Vector{Float64}, t::VAT)
    c = cut[t.output_index] * t.scalar_term.coefficient
    return SAT(c, t.scalar_term.variable)
end

function _add_constraint(
    cut::Vector{Float64},
    func::Union{VV, VAF},
    oa_opt::MOI.ModelLike,
    ::Nothing,
)
    cut_expr = _cut_expr(cut, func)
    return MOI.add_constraint(oa_opt, cut_expr, MOI.GreaterThan(0.0))
end

function _add_constraint(
    cut::Vector{Float64},
    func::Union{VV, VAF},
    oa_opt::MOI.ModelLike,
    cb,
)
    cut_expr = _cut_expr(cut, func)
    return MOI.submit(oa_opt, MOI.LazyConstraint(cb), cut_expr, MOI.GreaterThan(0.0))
end
