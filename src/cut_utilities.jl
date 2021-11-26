# cut utilities

function add_init_cuts(k::Int, opt::Optimizer)
    func = opt.con_funcs[k]
    cuts = Cuts.get_init_cuts(opt.con_sets[k])
    for cut in cuts
        _add_constraint(cut, func, opt.oa_opt, nothing)
    end
    opt.num_cuts += length(cuts)
    return nothing
end

function add_sep_cuts(k::Int, opt::Optimizer, cb = nothing)
    func = opt.con_funcs[k]
    point = MOIU.eval_variables(vi -> _get_val(vi, opt.oa_opt, cb), func)
    @assert !any(isnan, point)
    cuts = Cuts.get_sep_cuts(point, opt.con_sets[k], opt.tol_feas)
    for cut in cuts
        _add_constraint(cut, func, opt.oa_opt, cb)
    end
    opt.num_cuts += length(cuts)
    return !isempty(cuts)
end

function _get_val(vi::VI, oa_opt::MOI.ModelLike, ::Nothing)
    return MOI.get(oa_opt, MOI.VariablePrimal(), vi)
end

function _get_val(vi::VI, oa_opt::MOI.ModelLike, cb)
    return MOI.get(oa_opt, MOI.CallbackVariablePrimal(cb), vi)
end

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
