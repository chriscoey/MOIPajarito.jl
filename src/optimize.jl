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

    # optimizers
    oa_opt::Union{Nothing, MOI.AbstractOptimizer}
    conic_opt::Union{Nothing, MOI.AbstractOptimizer}

    # used by MOI wrapper
    obj_sense::MOI.OptimizationSense
    zeros_idxs::Vector{UnitRange{Int}} # TODO needed?

    # problem data
    c::Vector{Float64}
    A::SparseMatrixCSC{Float64, Int}
    b::Vector{Float64}
    G::SparseMatrixCSC{Float64, Int}
    h::Vector{Float64}
    obj_offset::Float64
    cones::Vector{MOI.AbstractVectorSet}
    cone_idxs::Vector{UnitRange{Int}}

    # models
    relax_model::JuMP.Model
    subp_model::JuMP.Model
    subp_vars::Vector{JuMP.VariableRef}
    oa_model::JuMP.Model
    oa_vars::Vector{JuMP.VariableRef}
    integer_vars::Vector{Int}
    oa_cones::Vector{Tuple{JuMP.ConstraintRef, Vector{JuMP.VariableRef}, MOI.AbstractVectorSet}}

    # modified throughout optimize and used after optimize
    status::MOI.TerminationStatusCode
    incumbent::Vector{Float64}
    obj_value::Float64
    obj_bound::Float64
    num_cuts::Int
    num_iters::Int
    num_callbacks::Int
    solve_time::Float64

    function Optimizer(
        verbose::Bool = true,
        tol_feas::Float64 = 1e-7,
        tol_rel_gap::Float64 = 1e-5,
        time_limit::Float64 = Inf,
        iteration_limit::Int = 1000,
        use_iterative_method::Union{Nothing, Bool} = nothing,
        oa_solver::Union{Nothing, MOI.OptimizerWithAttributes} = nothing,
        conic_solver::Union{Nothing, MOI.OptimizerWithAttributes} = nothing,
    )
        opt = new()
        opt.verbose = verbose
        opt.tol_feas = tol_feas
        opt.tol_rel_gap = tol_rel_gap
        opt.time_limit = time_limit
        opt.iteration_limit = iteration_limit
        opt.use_iterative_method = use_iterative_method
        opt.oa_solver = oa_solver
        opt.conic_solver = conic_solver
        opt.oa_opt = nothing
        opt.conic_opt = nothing
        return _empty_all(opt)
    end
end

function _empty_all(opt::Optimizer)
    opt.oa_vars = VI[]
    opt.subp_vars = VI[]
    opt.incumbent = Float64[]
    _empty_optimize(opt)
    return opt
end

function _empty_optimize(opt::Optimizer)
    opt.status = MOI.OPTIMIZE_NOT_CALLED
    opt.obj_value = NaN
    opt.obj_bound = NaN
    opt.num_cuts = 0
    opt.num_iters = 0
    opt.num_callbacks = 0
    opt.solve_time = NaN
    if !isnothing(opt.oa_opt)
        MOI.empty!(opt.oa_opt)
    end
    if !isnothing(opt.conic_opt)
        MOI.empty!(opt.conic_opt)
    end
    return opt
end

function _optimize!(opt::Optimizer)
    _start(opt)
    _setup_models(opt)

    # solve continuous relaxation
    relax_finish = solve_relaxation(opt)
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
    JuMP.optimize!(opt.oa_model)

    oa_status = JuMP.termination_status(opt.oa_model)
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
    opt.obj_bound = JuMP.objective_bound(opt.oa_model)

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
        # status = callback_node_status(cb, opt.oa_opt)
        # TODO
        # if status == MOI.CALLBACK_NODE_STATUS_FRACTIONAL
        #     # `callback_value(cb_data, x)` is not integer (to some tolerance).
        #     # If, for example, your lazy constraint generator requires an
        #     # integer-feasible primal solution, you can add a `return` here.
        #     return
        # elseif status == MOI.CALLBACK_NODE_STATUS_INTEGER
        #     # `callback_value(cb_data, x)` is integer (to some tolerance).
        # else
        #     @assert status == MOI.CALLBACK_NODE_STATUS_UNKNOWN
        #     # `callback_value(cb_data, x)` might be fractional or integer.
        # end
        # x_val = callback_value(cb_data, x)
        # if x_val > 2 + 1e-6
        #     con = @build_constraint(x <= 2)
        #     MOI.submit(model, MOI.LazyConstraint(cb_data), con)
        # end

        cuts_added = add_sep_cuts(opt, cb)
        if !cuts_added && opt.verbose
            println("no cuts were added during callback")
        end
    end
    MOI.set(opt.oa_opt, MOI.LazyConstraintCallback(), lazy_callback)

    if opt.verbose
        println("starting OA solver driven method")
    end
    JuMP.optimize!(opt.oa_model)
    oa_status = JuMP.termination_status(opt.oa_model)

    # TODO this should come from incumbent updated during lazy callbacks
    if oa_status == MOI.OPTIMAL
        opt.status = oa_status
        opt.obj_value = JuMP.objective_value(opt.oa_model)
        opt.obj_bound = JuMP.objective_bound(opt.oa_model)
        opt.incumbent = JuMP.value.(opt.oa_vars)
    else
        opt.status = oa_status
    end

    return
end

function solve_relaxation(opt::Optimizer)
    if opt.verbose
        println("solving continuous relaxation")
    end

    JuMP.optimize!(opt.relax_model)
    relax_status = JuMP.termination_status(opt.relax_model)

    if relax_status == MOI.OPTIMAL
        opt.obj_bound = JuMP.dual_objective_value(opt.relax_model)
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
            opt.obj_value = JuMP.objective_value(opt.relax_model)
            opt.incumbent = JuMP.value.(opt.subp_vars)
        end
        return true
    end

    return false
end

# solve subproblem with new integer variable bounds
# TODO handle non-equal bounds in lazy callback
function solve_subproblem(opt::Optimizer)
    # update integer bounds
    for i in opt.integer_vars
        x_i = opt.oa_vars[i]
        val = JuMP.value(x_i)
        @assert val ≈ round(val)
        # TODO map oa var to conic var
        JuMP.fix(opt.subp_vars[i], val; force = true)
    end

    # solve
    JuMP.optimize!(opt.relax_model)
    subp_status = JuMP.termination_status(opt.relax_model)

    if opt.verbose
        println("continuous subproblem status is $subp_status")
    end

    if subp_status == MOI.OPTIMAL
        obj_val = JuMP.objective_value(opt.relax_model)
        if _compare_obj(obj_val, opt.obj_value, opt)
            # update incumbent and objective value
            sol = JuMP.value.(opt.subp_vars)
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
    _conic_opt(opt)
    _oa_opt(opt)
    _empty_optimize(opt)
    opt.solve_time = time()

    if opt.obj_sense == MOI.MIN_SENSE
        opt.obj_value = Inf
    elseif opt.obj_sense == MOI.MAX_SENSE
        opt.obj_value = -Inf
    end
    opt.obj_bound = -opt.obj_value

    return nothing
end

# finalize and print
function _finish(opt::Optimizer)
    opt.solve_time = time() - opt.solve_time

    oa_status = MOI.get(opt.oa_opt, MOI.TerminationStatus())
    if opt.verbose && oa_status != MOI.OPTIMIZE_NOT_CALLED
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
    if opt.obj_sense == MOI.FEASIBILITY_SENSE
        return NaN
    end
    rel_gap = (opt.obj_value - opt.obj_bound) / (1e-5 + abs(opt.obj_value))
    if opt.obj_sense == MOI.MIN_SENSE
        return rel_gap
    else
        return -rel_gap
    end
end

# return true if objective value and bound are incompatible, else false
function _compare_obj(value::Float64, bound::Float64, opt::Optimizer)
    if opt.obj_sense == MOI.FEASIBILITY_SENSE
        return false
    elseif opt.obj_sense == MOI.MIN_SENSE
        return (value < bound)
    else
        return (value > bound)
    end
end

# setup conic and OA models
# TODO maybe just add directly in copy_to
function _setup_models(opt::Optimizer)
    # continuous relaxation model
    relax = opt.relax_model = JuMP.Model(() -> opt.conic_opt)
    x_relax = JuMP.@variable(relax, [1:length(opt.c)])
    JuMP.@objective(relax, Min, JuMP.dot(opt.c, x_relax))
    JuMP.@constraint(relax, opt.A * x_relax .== opt.b)

    # continuous subproblem model
    opt.subp_model = relax
    opt.subp_vars = x_relax
    # TODO delete integer vars
    # subp = opt.subp_model = JuMP.Model(() -> opt.conic_opt)
    # x_subp = JuMP.@variable(subp, [1:length(opt.c)])
    # opt.subp_vars = x_subp
    # JuMP.@objective(subp, Min, JuMP.dot(opt.c, x_subp))
    # JuMP.@constraint(subp, opt.A * x_subp .== opt.b)

    # mixed-integer OA model
    oa = opt.oa_model = JuMP.Model(() -> opt.oa_opt)
    x_oa = JuMP.@variable(oa, [1:length(opt.c)])
    opt.oa_vars = x_oa
    for i in opt.integer_vars
        JuMP.set_integer(x_oa[i])
    end
    JuMP.@objective(oa, Min, JuMP.dot(opt.c, x_oa))
    JuMP.@constraint(oa, opt.A * x_oa .== opt.b)

    # conic constraints
    opt.oa_cones = Tuple{JuMP.ConstraintRef, Vector{JuMP.VariableRef}, MOI.AbstractVectorSet}[]
    for (cone, idxs) in zip(opt.cones, opt.cone_idxs)
        h_i = opt.h[idxs]
        G_i = opt.G[idxs, :]
        K_relax_i = JuMP.@constraint(relax, h_i - G_i * x_relax in cone)
        if MOI.supports_constraint(opt.oa_opt, VAF, typeof(cone))
            JuMP.@constraint(oa, h_i - G_i * x_oa in cone)
        else
            s_i = JuMP.@variable(oa, [1:length(idxs)])
            JuMP.@constraint(oa, s_i .== h_i - G_i * x_oa)
            push!(opt.oa_cones, (K_relax_i, s_i, cone))
        end
    end
    return
end
