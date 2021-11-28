# outer approximation algorithm

mutable struct Optimizer <: MOI.AbstractOptimizer
    # options
    verbose::Bool
    tol_feas::Float64
    tol_rel_gap::Float64
    time_limit::Float64
    iteration_limit::Int
    use_iterative_method::Union{Nothing, Bool}
    oa_solver::Union{Nothing, MOI.OptimizerWithAttributes}
    conic_solver::Union{Nothing, MOI.OptimizerWithAttributes}

    # modified by MOI wrapper
    oa_opt::Union{Nothing, MOI.ModelLike}
    conic_opt::Union{Nothing, MOI.ModelLike}
    # TODO do we need both oa_vars and conic_vars?
    oa_vars::Vector{VI}
    conic_vars::Vector{VI}
    approx_types::Vector{Tuple}

    # modified throughout optimize and used after optimize
    status::MOI.TerminationStatusCode
    incumbent::Vector{Float64}
    obj_value::Float64
    obj_bound::Float64
    num_cuts::Int
    num_iters::Int
    num_callbacks::Int
    solve_time::Float64

    # temporary: set up at start of optimize and not used after optimize
    integer_vars::Vector{VI}
    approx_cons::Vector{Tuple{CI, Union{VV, VAF}, MOI.AbstractVectorSet}}

    function Optimizer()
        opt = new()
        opt.verbose = true
        opt.tol_feas = 1e-7
        opt.tol_rel_gap = 1e-5
        opt.time_limit = Inf
        opt.iteration_limit = 1000
        opt.use_iterative_method = nothing
        opt.oa_solver = nothing
        opt.conic_solver = nothing
        return _empty_all(opt)
    end
end

function _empty_all(opt::Optimizer)
    opt.oa_opt = nothing
    opt.conic_opt = nothing
    opt.oa_vars = VI[]
    opt.conic_vars = VI[]
    opt.approx_types = Tuple[]
    opt.incumbent = Float64[]
    _empty_optimize(opt)
    return opt
end

function _empty_optimize(opt::Optimizer)
    opt.status = MOI.OPTIMIZE_NOT_CALLED
    fill!(opt.incumbent, NaN)
    opt.obj_value = NaN
    opt.obj_bound = NaN
    opt.num_cuts = 0
    opt.num_iters = 0
    opt.num_callbacks = 0
    opt.solve_time = NaN
    return opt
end

function MOI.optimize!(opt::Optimizer)
    _start(opt)

    # solve continuous relaxation
    relax_finish = solve_relaxation(opt)
    if relax_finish
        _finish(opt)
        return
    end

    # add continuous relaxation cuts and initial fixed cuts
    add_subp_cuts(opt)
    # add_init_cuts(opt)

    if opt.use_iterative_method
        # solve using iterative method
        while true
            opt.num_iters += 1
            finish_iter = iterative_method(opt)
            finish_iter && break
        end
    else
        # solve using OA solver driven method
        oa_solver_driven_method(opt)
    end

    _finish(opt)
    return
end

# one iteration of the iterative method
function iterative_method(opt::Optimizer)
    MOI.optimize!(opt.oa_opt)

    oa_status = MOI.get(opt.oa_opt, MOI.TerminationStatus())
    if oa_status == MOI.INFEASIBLE
        if opt.verbose
            println("infeasibility detected while iterating; terminating")
        end
        opt.status = oa_status
        return true
    elseif oa_status != MOI.OPTIMAL
        @warn("OA solver status $oa_status is not handled")
        opt.status = MOI.OTHER_ERROR
        return true
    end

    # update objective bound
    opt.obj_bound = MOI.get(opt.oa_opt, MOI.ObjectiveBound())

    # solve conic subproblem with fixed integer solution and update incumbent
    subp_finish = solve_subproblem(opt)
    subp_finish && return true

    # print and check convergence
    obj_rel_gap = _obj_rel_gap(opt)
    if opt.verbose
        Printf.@printf(
            "%5d %8d %12.4e %12.4e %12.4e\n",
            opt.num_iters,
            opt.num_cuts,
            opt.obj_value,
            opt.obj_bound,
            obj_rel_gap
        )
    end
    if !isnan(obj_rel_gap) && obj_rel_gap < opt.tol_rel_gap
        if opt.verbose
            println("objective relative gap $obj_rel_gap reached; terminating")
        end
        opt.status = MOI.OPTIMAL
        return true
    end

    # check iteration limit
    if opt.num_iters == opt.iteration_limit
        if opt.verbose
            println("iteration limit ($(opt.num_iters)) reached; terminating")
        end
        opt.status = MOI.ITERATION_LIMIT
        return true
    end

    # check time limit
    if time() - opt.solve_time >= opt.time_limit
        if opt.verbose
            println("time limit ($(opt.time_limit)) reached; terminating")
        end
        opt.status = MOI.TIME_LIMIT
        return true
    end

    # add OA cuts
    subp_cuts_added = add_subp_cuts(opt)
    if !subp_cuts_added
        if opt.verbose
            println("no subproblem cuts were added during an iteration")
        end
        # try separation cuts
        cuts_added = add_sep_cuts(opt)
        if !cuts_added
            if opt.verbose
                println("no cuts were added during an iteration; terminating")
            end
            # TODO check if almost optimal
            opt.status = MOI.OTHER_ERROR
            return true
        end
    end

    return false
end

# OA solver driven method using callbacks
function oa_solver_driven_method(opt::Optimizer)
    function lazy_callback(cb)
        opt.num_callbacks += 1
        cuts_added = add_sep_cuts(opt, cb)
        if !cuts_added && opt.verbose
            println("no cuts were added during callback")
        end
    end
    MOI.set(opt.oa_opt, MOI.LazyConstraintCallback(), lazy_callback)

    if opt.verbose
        println("starting OA solver driven method")
    end
    MOI.optimize!(opt.oa_opt)
    oa_status = MOI.get(opt.oa_opt, MOI.TerminationStatus())

    # TODO this should come from incumbent updated during lazy callbacks
    if oa_status == MOI.OPTIMAL
        opt.status = oa_status
        opt.obj_value = MOI.get(opt.oa_opt, MOI.ObjectiveValue())
        opt.obj_bound = MOI.get(opt.oa_opt, MOI.ObjectiveBound())
        opt.incumbent = MOI.get(opt.oa_opt, MOI.VariablePrimal(), opt.oa_vars)
    else
        opt.status = oa_status
    end

    return
end

function solve_relaxation(opt::Optimizer)
    if opt.verbose
        println("solving continuous relaxation")
    end

    MOI.optimize!(opt.conic_opt)
    relax_status = MOI.get(opt.conic_opt, MOI.TerminationStatus())

    if relax_status == MOI.OPTIMAL
        opt.obj_bound = MOI.get(opt.conic_opt, MOI.DualObjectiveValue())
        if opt.verbose
            println(
                "continuous relaxation status is $relax_status " *
                "with dual objective value $(opt.obj_bound)",
            )
        end
    elseif relax_status == MOI.DUAL_INFEASIBLE
        if opt.verbose
            println("continuous relaxation status is $relax_status; Pajarito may fail")
        end
    elseif relax_status == MOI.INFEASIBLE
        if opt.verbose
            println("infeasibility detected from continuous relaxation; terminating")
        end
        opt.status = relax_status
        return true
    else
        @warn("OA solver status $relax_status is not handled")
        opt.status = MOI.OTHER_ERROR
        return true
    end

    if isempty(opt.integer_vars)
        # problem is continuous
        if opt.verbose
            println("problem is continuous; terminating without using OA solver")
        end
        if relax_status == MOI.OPTIMAL
            opt.status = relax_status
            opt.obj_value = MOI.get(opt.conic_opt, MOI.ObjectiveValue())
            opt.incumbent = MOI.get(opt.conic_opt, MOI.VariablePrimal(), opt.conic_vars)
        end
        return true
    end

    return false
end

# solve subproblem with new integer variable bounds
# TODO handle non-equal bounds in lazy callback
function solve_subproblem(opt::Optimizer)
    # update integer bounds
    for vi in opt.integer_vars
        val = MOI.get(opt.oa_opt, MOI.VariablePrimal(), vi)
        @assert val ≈ round(val)
        @show vi
        @show val
        @show MOI.get(opt.conic_opt, MOI.ListOfConstraintTypesPresent())
        # @show MOI.is_valid(opt.conic_opt, CI{VI, MOI.LessThan{Float64}}())
        # @show MOI.is_valid(opt.conic_opt, CI{VI, MOI.Interval{Float64}}())
        # MOIU._print_model(opt.conic_opt)
        # println()
        # TODO map oa var to conic var
        old_val = MOI.get(opt.conic_opt, MOI.VariablePrimal(), vi)
        @show old_val
        MOI.add_constraint(opt.conic_opt, vi, MOI.EqualTo{Float64}(val))
    end

    # solve
    MOI.optimize!(opt.conic_opt)
    subp_status = MOI.get(opt.conic_opt, MOI.TerminationStatus())

    if opt.verbose
        println("continuous subproblem status is $subp_status")
    end

    if subp_status == MOI.OPTIMAL
        obj_val = MOI.get(opt.conic_opt, MOI.ObjectiveValue())
        if _compare_obj(obj_val, opt.obj_value, opt)
            # update incumbent and objective value
            sol = MOI.get(opt.conic_opt, MOI.VariablePrimal(), opt.conic_vars)
            copyto!(opt.incumbent, sol)
            opt.obj_value = obj_val
        end
        return false
    elseif subp_status == MOI.INFEASIBLE
        return false
    end

    @warn("OA solver status $subp_status is not handled")
    opt.status = MOI.OTHER_ERROR
    return true
end

# initialize and print
function _start(opt::Optimizer)
    @assert !isnothing(opt.oa_opt)
    @assert !isnothing(opt.conic_opt)
    _empty_optimize(opt)

    sense = MOI.get(opt.oa_opt, MOI.ObjectiveSense())
    if sense == MOI.MIN_SENSE
        opt.obj_value = Inf
    elseif sense == MOI.MAX_SENSE
        opt.obj_value = -Inf
    end
    opt.obj_bound = -opt.obj_value

    # integer variables
    cis = MOI.get(opt.oa_opt, MOI.ListOfConstraintIndices{VI, MOI.Integer}())
    opt.integer_vars = VI[VI(ci.value) for ci in cis]

    # integer variable bound constraints
    # TODO use floor and ceil to tighten bounds for conic subproblems
    # int_bounds = [_int_bounds(vi, opt.conic_opt) for vi in opt.integer_vars]

    # constraints needing outer approximation
    opt.approx_cons = Tuple{CI, Union{VV, VAF}, MOI.AbstractVectorSet}[]
    FSs = opt.approx_types
    for (F, S) in FSs, ci in MOI.get(opt.conic_opt, MOI.ListOfConstraintIndices{F, S}())
        func = MOI.get(opt.conic_opt, MOI.ConstraintFunction(), ci)
        set = MOI.get(opt.conic_opt, MOI.ConstraintSet(), ci)
        push!(opt.approx_cons, (ci, func, set))
    end

    opt.solve_time = time()

    return nothing
end

# finalize and print
function _finish(opt::Optimizer)
    opt.solve_time = time() - opt.solve_time

    if opt.verbose
        oa_status = MOI.get(opt.oa_opt, MOI.TerminationStatus())
        println("OA solver finished with status $oa_status, after $(opt.num_cuts) cuts")
        if opt.use_iterative_method
            println("iterative method used $(opt.num_iters) iterations\n")
        else
            println("OA solver driven method used $(opt.num_callbacks) callbacks\n")
        end
    end

    return nothing
end

# compute objective relative gap
# TODO decide whether to use 1e-5 constant
function _obj_rel_gap(opt::Optimizer)
    sense = MOI.get(opt.oa_opt, MOI.ObjectiveSense())
    if sense == MOI.FEASIBILITY_SENSE
        return NaN
    end
    rel_gap = (opt.obj_value - opt.obj_bound) / (1e-5 + abs(opt.obj_value))
    if sense == MOI.MIN_SENSE
        return rel_gap
    else
        return -rel_gap
    end
end

# return true if objective value and bound are incompatible, else false
function _compare_obj(value::Float64, bound::Float64, opt::Optimizer)
    sense = MOI.get(opt.oa_opt, MOI.ObjectiveSense())
    if sense == MOI.FEASIBILITY_SENSE
        return false
    elseif sense == MOI.MIN_SENSE
        return (value < bound)
    else
        return (value > bound)
    end
end

# function _int_bounds(vi::VI, conic_opt::MOI.ModelLike)
#     ci_interval = CI{VI, MOI.Interval{Float64}}(vi)
#     if MOI.is_valid(conic_opt, )

#     return ci
# end
