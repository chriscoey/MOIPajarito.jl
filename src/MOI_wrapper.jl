#=
MathOptInterface wrapper of Pajarito solver
=#

_is_empty(::Nothing) = true
_is_empty(opt::MOI.AbstractOptimizer) = MOI.is_empty(opt)

MOI.is_empty(opt::Optimizer) = (_is_empty(opt.oa_opt) && _is_empty(opt.conic_opt))

MOI.empty!(opt::Optimizer) = _empty_all(opt)

MOI.get(::Optimizer, ::MOI.SolverName) = "Pajarito"

MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.set(opt::Optimizer, ::MOI.Silent, value::Bool)
    opt.verbose = !value
    return
end

MOI.get(opt::Optimizer, ::MOI.Silent) = !opt.verbose

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true

function MOI.set(opt::Optimizer, ::MOI.TimeLimitSec, value::Union{Real, Nothing})
    opt.time_limit = something(value, Inf)
    return
end

function MOI.get(opt::Optimizer, ::MOI.TimeLimitSec)
    if isfinite(opt.time_limit)
        return opt.time_limit
    end
    return
end

function MOI.get(opt::Optimizer, ::MOI.SolveTimeSec)
    if isnan(opt.solve_time)
        error("solve has not been called")
    end
    return opt.solve_time
end

MOI.get(opt::Optimizer, ::MOI.RawStatusString) = string(opt.status)

function MOI.set(opt::Optimizer, param::MOI.RawOptimizerAttribute, value)
    return setproperty!(opt, Symbol(param.name), value)
end

function MOI.get(opt::Optimizer, param::MOI.RawOptimizerAttribute)
    return getproperty(opt, Symbol(param.name))
end

function MOI.supports(
    ::Optimizer,
    ::Union{MOI.ObjectiveSense, MOI.ObjectiveFunction{<:Union{VI, SAF}}},
)
    return true
end

function MOI.supports_constraint(
    opt::Optimizer,
    F::Type{VI},
    S::Type{MOI.Integer},
)
    return MOI.supports_constraint(_oa_opt(opt), F, S)
end

function MOI.supports_constraint(
    opt::Optimizer,
    F::Type{<:Union{VV, VAF}},
    S::Type{<:Union{MOI.Zeros, MOI.AbstractVectorSet}},
)
    return MOI.supports_constraint(_conic_opt(opt), F, S)
end

MOI.supports(::Optimizer, ::MOI.VariablePrimalStart, ::Type{VI}) = true

function MOI.set(opt::Optimizer, attr::MOI.VariablePrimalStart, vi::VI, value)
    opt.incumbent[vi.value] = something(value, NaN)
    error("VariablePrimalStart not implemented")
    return
end

function MOI.copy_to(opt::Optimizer, src::MOI.ModelLike)
    idx_map = MOI.Utilities.IndexMap()

    # variables
    n = MOI.get(src, MOI.NumberOfVariables()) # columns of A
    j = 0
    for vj in MOI.get(src, MOI.ListOfVariableIndices())
        j += 1
        idx_map[vj] = VI(j)
    end
    @assert j == n
    for attr in MOI.get(src, MOI.ListOfVariableAttributesSet())
        if attr == MOI.VariableName()
            continue
        end
        throw(MOI.UnsupportedAttribute(attr))
    end

    # integer variables
    cis = MOI.get(src, MOI.ListOfConstraintIndices{VI, MOI.Integer}())
    opt.integer_vars = VI[VI(ci.value) for ci in cis]

    # objective
    opt.obj_sense = MOI.MIN_SENSE
    model_c = zeros(n)
    obj_offset = 0.0
    for attr in MOI.get(src, MOI.ListOfModelAttributesSet())
        if attr == MOI.Name()
            continue
        elseif attr == MOI.ObjectiveSense()
            opt.obj_sense = MOI.get(src, MOI.ObjectiveSense())
        elseif attr isa MOI.ObjectiveFunction
            F = MOI.get(src, MOI.ObjectiveFunctionType())
            if !(F <: Union{VI, SAF})
                error("objective function type $F not supported")
            end
            obj = convert(SAF, MOI.get(src, MOI.ObjectiveFunction{F}()))
            for t in obj.terms
                model_c[idx_map[t.variable].value] += t.coefficient
            end
            obj_offset = obj.constant
        else
            throw(MOI.UnsupportedAttribute(attr))
        end
    end
    if opt.obj_sense == MOI.MAX_SENSE
        model_c .*= -1
        obj_offset *= -1
    end
    opt.obj_offset = obj_offset
    opt.c = model_c

    # constraints
    get_src_cons(F, S) = MOI.get(src, MOI.ListOfConstraintIndices{F, S}())
    get_con_fun(con_idx) = MOI.get(src, MOI.ConstraintFunction(), con_idx)
    get_con_set(con_idx) = MOI.get(src, MOI.ConstraintSet(), con_idx)

    # equality constraints
    (IA, JA, VA) = (Int[], Int[], Float64[])
    model_b = Float64[]
    opt.zeros_idxs = zeros_idxs = Vector{UnitRange{Int}}()
    for F in (VV, VAF), ci in get_src_cons(F, MOI.Zeros)
        fi = get_con_fun(ci)
        _con_IJV(IA, JA, VA, model_b, zeros_idxs, fi, idx_map)
        idx_map[ci] = ci
    end
    opt.A = dropzeros!(sparse(IA, JA, VA, length(model_b), n))
    opt.b = model_b

    # conic constraints
    (IG, JG, VG) = (Int[], Int[], Float64[])
    model_h = Float64[]
    cones = MOI.AbstractVectorSet[]
    cone_idxs = Vector{UnitRange{Int}}()

    # build up one nonnegative cone
    for F in (VV, VAF), ci in get_src_cons(F, MOI.Nonnegatives)
        fi = get_con_fun(ci)
        _con_IJV(IG, JG, VG, model_h, fi, idx_map)
        idx_map[ci] = ci
    end
    if !isempty(model_h)
        q = length(model_h)
        push!(cones, MOI.Nonnegatives(q))
        push!(cone_idxs, 1:q)
    end

    # other conic constraints
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        if !MOI.supports_constraint(opt, F, S)
            throw(MOI.UnsupportedConstraint{F, S}())
        end
        for attr in MOI.get(src, MOI.ListOfConstraintAttributesSet{F, S}())
            if attr == MOI.ConstraintName() ||
               attr == MOI.ConstraintPrimalStart() ||
               attr == MOI.ConstraintDualStart()
                continue
            end
            throw(MOI.UnsupportedAttribute(attr))
        end
        if S == MOI.Zeros || S == MOI.Nonnegatives || S == MOI.Integer
            continue # already copied these constraints
        end

        for ci in get_src_cons(F, S)
            fi = get_con_fun(ci)
            idxs = _con_IJV(IG, JG, VG, model_h, fi, idx_map)
            push!(cone_idxs, idxs)
            si = get_con_set(ci)
            push!(cones, si)
            idx_map[ci] = ci
        end
    end

    opt.G = dropzeros!(sparse(IG, JG, VG, length(model_h), length(model_c)))
    opt.h = model_h
    opt.cones = cones
    opt.cone_idxs = cone_idxs

    return idx_map
end

MOI.optimize!(opt::Optimizer) = _optimize!(opt)

MOI.get(opt::Optimizer, ::MOI.TerminationStatus) = opt.status

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

_sense_val(sense::MOI.OptimizationSense) = (sense == MOI.MAX_SENSE ? -1 : 1)

function MOI.get(opt::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(opt, attr)
    return _sense_val(opt.obj_sense) * opt.obj_value
end

function MOI.get(opt::Optimizer, attr::MOI.ObjectiveBound)
    MOI.check_result_index_bounds(opt, attr)
    return _sense_val(opt.obj_sense) * opt.obj_bound
end

MOI.get(opt::Optimizer, ::MOI.ResultCount) = 1

function MOI.get(opt::Optimizer, attr::MOI.VariablePrimal, vi::VI)
    MOI.check_result_index_bounds(opt, attr)
    return opt.incumbent[vi.value]
end

function _con_IJV(
    IM::Vector{Int},
    JM::Vector{Int},
    VM::Vector,
    vect::Vector,
    func::VV,
    idx_map::MOI.IndexMap,
)
    dim = MOI.output_dimension(func)
    idxs = length(vect) .+ (1:dim)
    append!(vect, 0.0 for _ in 1:dim)
    append!(IM, idxs)
    append!(JM, idx_map[vi].value for vi in func.variables)
    append!(VM, -1.0 for _ in 1:dim)
    return idxs
end

function _con_IJV(
    IM::Vector{Int},
    JM::Vector{Int},
    VM::Vector,
    vect::Vector,
    func::VAF,
    idx_map::MOI.IndexMap,
)
    dim = MOI.output_dimension(func)
    start = length(vect)
    append!(vect, func.constants)
    append!(IM, start + vt.output_index for vt in func.terms)
    append!(JM, idx_map[vt.scalar_term.variable].value for vt in func.terms)
    append!(VM, -vt.scalar_term.coefficient for vt in func.terms)
    return start .+ (1:dim)
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
