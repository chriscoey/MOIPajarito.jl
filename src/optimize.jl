# outer approximation algorithm

mutable struct Optimizer <: MOI.AbstractOptimizer
    # options
    verbose::Bool
    tol_feas::Float64
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
        opt.time_limit = Inf
        opt.iteration_limit = 1000
        opt.use_iterative_method = nothing
        opt.oa_solver = nothing
        opt.conic_solver = nothing
        return _empty(opt)
    end
end

function _empty(opt::Optimizer)
    opt.oa_opt = nothing
    opt.conic_opt = nothing
    opt.oa_vars = VI[]
    opt.conic_vars = VI[]
    opt.approx_types = Tuple[]

    opt.incumbent = Float64[]
    opt.obj_value = NaN
    opt.obj_bound = NaN

    opt.status = MOI.OPTIMIZE_NOT_CALLED
    opt.num_cuts = 0
    opt.num_iters = 0
    opt.num_callbacks = 0
    opt.solve_time = NaN

    return opt
end

function MOI.optimize!(opt::Optimizer)
    _start(opt)

    # solve continuous relaxation
    relax_finish = solve_relax(opt)
    if relax_finish
        _finish(opt)
        return
    end

    # add continuous relaxation cuts and initial fixed cuts
    add_subp_cuts(opt)
    add_init_cuts(opt)

    if opt.use_iterative_method
        # solve using iterative method
        while true
            iterative_method(opt) && break
        end
    else
        # solve using OA solver driven method
        oa_solver_driven_method(opt)
    end

    _finish(opt)
    return
end

# initialize and print
function _start(opt::Optimizer)
    @assert !isnothing(opt.oa_opt)
    @assert !isnothing(opt.conic_opt)

    # integer variables
    cis = MOI.get(opt.oa_opt, MOI.ListOfConstraintIndices{VI, MOI.Integer}())
    # opt.integer_vars = VI[MOI.get(opt.oa_opt, MOI.ConstraintFunction(), ci) for ci in cis]
    opt.integer_vars = VI[ci.value for ci in cis]

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

function solve_relax(opt::Optimizer)
    if opt.verbose
        println("solving continuous relaxation")
    end

    MOI.optimize!(opt.conic_opt)
    relax_status = MOI.get(opt.conic_opt, MOI.TerminationStatus())

    # TODO maybe dispatch on status rather than if-else
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

# one iteration of the iterative method
function iterative_method(opt::Optimizer)
    MOI.optimize!(opt.oa_opt)

    # TODO maybe dispatch on status rather than if-else
    oa_status = MOI.get(opt.oa_opt, MOI.TerminationStatus())
    if oa_status == MOI.OPTIMAL
        obj_bound = MOI.get(opt.oa_opt, MOI.ObjectiveBound())
        if opt.verbose
            Printf.@printf("%5d %8d %12.4e\n", opt.num_iters, opt.num_cuts, obj_bound)
        end
    elseif oa_status == MOI.INFEASIBLE
        if opt.verbose
            println("infeasibility detected while iterating; terminating")
        end
        opt.status = oa_status
        return true
        # elseif oa_status == MOI.INFEASIBLE_OR_UNBOUNDED
        #     if opt.verbose
        #         println("OA solver status is $oa_status; assuming infeasibility")
        #     end
        #     opt.status = MOI.INFEASIBLE
        #     return true
    else
        @warn("OA solver status $oa_status is not handled")
        opt.status = MOI.OTHER_ERROR
        return true
    end

    if opt.num_iters == opt.iteration_limit
        if opt.verbose
            println("iteration limit ($(opt.num_iters)) reached; terminating")
        end
        opt.status = :ITERATION_LIMIT
        return true
    end

    if time() - opt.solve_time >= opt.time_limit
        opt.verbose && println("time limit ($(opt.time_limit)) reached; terminating")
        opt.status = :TIME_LIMIT
        return true
    end

    cuts_added = add_sep_cuts(opt)
    if !cuts_added
        if opt.verbose
            println("no cuts were added; terminating")
        end
        opt.status = MOI.OPTIMAL
        return true
    end

    opt.num_iters += 1
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
    return
end
