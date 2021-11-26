# MathOptInterface wrapper

_is_empty(opt) = (isnothing(opt) || MOI.is_empty(opt))
MOI.is_empty(opt::Optimizer) = (_is_empty(opt.oa_opt) && _is_empty(opt.conic_opt))

MOI.empty!(opt::Optimizer) = _empty(opt)

MOI.get(::Optimizer, ::MOI.SolverName) = "Pajarito"

MOI.supports_incremental_interface(::Optimizer) = true

MOI.copy_to(opt::Optimizer, src::MOI.ModelLike) = MOIU.default_copy_to(opt, src)

function MOI.add_variable(opt::Optimizer)
    push!(opt.oa_vars, MOI.add_variable(_oa_opt(opt)))
    push!(opt.conic_vars, MOI.add_variable(_conic_opt(opt)))
    push!(opt.incumbent, NaN)
    return VI(length(opt.oa_vars))
end

function MOI.add_variables(opt::Optimizer, n::Int)
    old_n = length(opt.oa_vars)
    append!(opt.oa_vars, MOI.add_variables(_oa_opt(opt), n))
    append!(opt.conic_vars, MOI.add_variables(_conic_opt(opt), n))
    append!(opt.incumbent, fill(NaN, n))
    return VI.(old_n .+ (1:n))
end

MOI.get(opt::Optimizer, ::MOI.NumberOfVariables) = length(opt.incumbent)

MOI.supports(::Optimizer, ::MOI.VariablePrimalStart, ::Type{VI}) = true

function MOI.set(opt::Optimizer, attr::MOI.VariablePrimalStart, vi::VI, value)
    opt.incumbent[vi.value] = something(value, NaN)
    return
end

# only discrete set supported is integer, because we want bridges to add explicit
# variable bounds for e.g. ZeroOne so that these are available to the conic solver
function MOI.supports_constraint(opt::Optimizer, F::Type{VI}, S::Type{MOI.Integer})
    return MOI.supports_constraint(_oa_opt(opt), F, S)
end

function MOI.add_constraint(opt::Optimizer, vi::VI, set::MOI.Integer)
    return MOI.add_constraint(_oa_opt(opt), _map(opt.oa_vars, vi), set)
end

# MOI.is_valid(opt::Optimizer, ci::CI{VI, MOI.Integer}) = MOI.is_valid(_oa_opt(opt), ci)

MOI.delete(opt::Optimizer, ci::CI{VI, MOI.Integer}) = MOI.delete(_oa_opt(opt), ci)

function MOI.supports_constraint(
    opt::Optimizer,
    F::Type{<:MOI.AbstractFunction},
    S::Type{<:MOI.AbstractSet},
)
    oa_supports = MOI.supports_constraint(_oa_opt(opt), F, S)
    conic_supports = MOI.supports_constraint(_conic_opt(opt), F, S)
    return oa_supports && conic_supports
end

function MOI.add_constraint(
    opt::Optimizer,
    func::MOI.AbstractFunction,
    set::MOI.AbstractSet,
)
    MOI.add_constraint(_conic_opt(opt), _map(opt.conic_vars, func), set)
    return MOI.add_constraint(_oa_opt(opt), _map(opt.oa_vars, func), set)
end

function MOI.supports_constraint(
    opt::Optimizer,
    F::Type{<:Union{VV, VAF}},
    S::Type{<:MOI.AbstractVectorSet},
)
    return MOI.supports_constraint(_conic_opt(opt), F, S)
end

function MOI.add_constraint(
    opt::Optimizer,
    func::F,
    set::S,
) where {F <: Union{VV, VAF}, S <: MOI.AbstractVectorSet}
    func = MOIU.canonical(func)
    ci = MOI.add_constraint(_conic_opt(opt), _map(opt.conic_vars, func), set)
    if MOI.supports_constraint(_oa_opt(opt), F, S)
        MOI.add_constraint(_oa_opt(opt), _map(opt.oa_vars, func), set)
    else
        # (F, S) constraints need outer approximation
        push!(opt.approx_types, (F, S))
    end
    return ci
end

# MOI.is_valid(opt::Optimizer, ci::CI) = MOI.is_valid(_conic_opt(opt), ci)

function MOI.delete(opt::Optimizer, ci::CI)
    MOI.delete(_conic_opt(opt), ci)
    if MOI.is_valid(_oa_opt(opt), ci)
        MOI.delete(_oa_opt(opt), ci)
    end
    return nothing
end

_map(vars::Vector{VI}, vis) = MOIU.map_indices(vi -> vars[vi.value], vis)

MOI.get(opt::Optimizer, attr::MOI.NumberOfConstraints) = MOI.get(_conic_opt(model), attr)

function MOI.get(opt::Optimizer, attr::MOI.NumberOfConstraints{VI, MOI.Integer})
    return MOI.get(_oa_opt(model), attr)
end

MOI.supports(opt::Optimizer, attr::MOI.ObjectiveSense) = true

function MOI.set(opt::Optimizer, attr::MOI.ObjectiveSense, sense)
    MOI.set(_oa_opt(opt), attr, sense)
    MOI.set(_conic_opt(opt), attr, sense)
    return
end

MOI.get(opt::Optimizer, attr::MOI.ObjectiveSense) = MOI.get(_oa_opt(opt), attr)

function MOI.supports(opt::Optimizer, attr::MOI.ObjectiveFunction)
    oa_supports = MOI.supports(_oa_opt(opt), attr)
    conic_supports = MOI.supports(_conic_opt(opt), attr)
    return oa_supports && conic_supports
end

function MOI.set(opt::Optimizer, attr::MOI.ObjectiveFunction, func)
    MOI.set(_oa_opt(opt), attr, _map(opt.oa_vars, func))
    MOI.set(_conic_opt(opt), attr, _map(opt.conic_vars, func))
    return
end

function MOI.get(opt::Optimizer, param::MOI.RawOptimizerAttribute)
    return getproperty(opt, Symbol(param.name))
end

function MOI.supports(::Optimizer, param::MOI.RawOptimizerAttribute)
    return (Symbol(param.name) in fieldnames(Optimizer))
end

function MOI.set(opt::Optimizer, param::MOI.RawOptimizerAttribute, value)
    setproperty!(opt, Symbol(param.name), value)
    return
end

MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.set(opt::Optimizer, ::MOI.Silent, value::Bool)
    opt.verbose = !value
    return
end

MOI.get(opt::Optimizer, ::MOI.Silent) = !opt.verbose

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true

function MOI.set(opt::Optimizer, ::MOI.TimeLimitSec, value::Nothing)
    MOI.set(opt, MOI.RawOptimizerAttribute("time_limit"), Inf)
    return
end

function MOI.set(opt::Optimizer, ::MOI.TimeLimitSec, value)
    MOI.set(opt, MOI.RawOptimizerAttribute("time_limit"), value)
    return
end

function MOI.get(opt::Optimizer, ::MOI.TimeLimitSec)
    value = MOI.get(opt, MOI.RawOptimizerAttribute("time_limit"))
    return (isfinite(value) ? value : nothing)
end

MOI.get(opt::Optimizer, ::MOI.SolveTimeSec) = opt.solve_time

MOI.get(opt::Optimizer, ::MOI.TerminationStatus) = opt.status

MOI.get(opt::Optimizer, ::MOI.RawStatusString) = string(opt.status)

function MOI.get(opt::Optimizer, attr::MOI.VariablePrimal, vi::VI)
    MOI.check_result_index_bounds(opt, attr)
    return opt.incumbent[vi.value]
end

function MOI.get(opt::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(opt, attr)
    return opt.obj_value
end

MOI.get(opt::Optimizer, ::MOI.ObjectiveBound) = opt.obj_bound

function MOI.get(opt::Optimizer, attr::MOI.PrimalStatus)
    if attr.result_index != 1
        return MOI.NO_SOLUTION
    end
    term_status = MOI.get(opt, MOI.TerminationStatus())
    if term_status == MOI.OPTIMAL
        return MOI.FEASIBLE_POINT
    elseif term_status == MOI.ALMOST_OPTIMAL
        return MOI.NEARLY_FEASIBLE_POINT
    end
    return MOI.NO_SOLUTION
end

MOI.get(::Optimizer, ::MOI.DualStatus) = MOI.NO_SOLUTION

function MOI.get(opt::Optimizer, ::MOI.ResultCount)
    return Int(MOI.get(opt, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT)
end

function _oa_opt(opt::Optimizer)
    if isnothing(opt.oa_opt)
        if isnothing(opt.oa_solver)
            error("No outer approximation solver specified (set `oa_solver`)")
        end
        opt.oa_opt = MOI.instantiate(opt.oa_solver, with_bridge_type = Float64)
        # check whether lazy constraints are supported
        supports_lazy = MOI.supports(opt.oa_opt, MOI.LazyConstraintCallback())
        if isnothing(opt.use_iterative_method)
            # default to OA-solver-driven method if possible
            opt.use_iterative_method = !supports_lazy
        elseif !opt.use_iterative_method && !supports_lazy
            error(
                "Outer approximation solver (`oa_solver`) does not support " *
                "lazy constraint callbacks (`use_iterative_method` must be `true`)",
            )
        end
    end
    return opt.oa_opt
end

function _conic_opt(opt::Optimizer)
    if isnothing(opt.conic_opt)
        if isnothing(opt.conic_solver)
            error("No primal-dual conic solver specified (set `conic_solver`)")
        end
        opt.conic_opt = MOI.instantiate(opt.conic_solver, with_bridge_type = Float64)
    end
    return opt.conic_opt
end
