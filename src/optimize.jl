# outer approximation algorithm

mutable struct Optimizer <: MOI.AbstractOptimizer
    verbose::Bool
    tol_feas::Float64
    time_limit::Float64
    iteration_limit::Int
    use_iterative_method::Union{Nothing, Bool}
    oa_solver::Union{Nothing, MOI.OptimizerWithAttributes}
    conic_solver::Union{Nothing, MOI.OptimizerWithAttributes}

    oa_opt::Union{Nothing, MOI.ModelLike}
    conic_opt::Union{Nothing, MOI.ModelLike}

    oa_vars::Vector{VI}
    conic_vars::Vector{VI}
    int_indices::BitSet

    con_funcs::Vector{Union{VV, VAF}}
    con_sets::Vector{MOI.AbstractVectorSet}

    incumbent::Vector{Float64}
    obj_value::Float64
    obj_bound::Float64

    status::MOI.TerminationStatusCode
    num_cuts::Int
    num_iters::Int
    num_callbacks::Int
    solve_time::Float64

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
    opt.int_indices = BitSet()

    opt.con_funcs = Vector{Union{VV, VAF}}[]
    opt.con_sets = MOI.AbstractVectorSet[]

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
    opt.solve_time = time()

    # TODO solve continuous relaxation and add cuts

    # add initial fixed cuts
    for k in eachindex(opt.con_sets)
        add_init_cuts(k, opt)
    end
    if opt.verbose
        println("added $(opt.num_cuts) initial cuts; starting OA solver")
    end

    if opt.use_iterative_method
        iterative_method(opt)
    else
        oa_solver_driven_method(opt)
    end

    if opt.verbose
        oa_status = MOI.get(opt.oa_opt, MOI.TerminationStatus())
        println("OA solver finished with status $oa_status, after $(opt.num_cuts) cuts")
        if opt.use_iterative_method
            println("iterative method used $(opt.num_iters) iterations\n")
        else
            println("OA solver driven method used $(opt.num_callbacks) callbacks\n")
        end
    end

    opt.solve_time = time() - opt.solve_time
    return
end

# TODO handle statuses properly
function iterative_method(opt::Optimizer)
    while true
        MOI.optimize!(opt.oa_opt)

        oa_opt_status = MOI.get(opt.oa_opt, MOI.TerminationStatus())
        if oa_opt_status == MOI.OPTIMAL
            obj_bound = MOI.get(opt.oa_opt, MOI.ObjectiveBound())
            if opt.verbose
                Printf.@printf("%5d %8d %12.4e\n", opt.num_iters, opt.num_cuts, obj_bound)
            end
        elseif oa_opt_status == MOI.INFEASIBLE
            if opt.verbose
                println("infeasibility detected; terminating")
            end
            opt.status = oa_opt_status
            break
        elseif oa_opt_status == MOI.INFEASIBLE_OR_UNBOUNDED
            if opt.verbose
                println("OA solver status is $oa_opt_status; assuming infeasibility")
            end
            opt.status = MOI.INFEASIBLE
            break
        else
            @warn("OA solver status $oa_opt_status is not handled")
            opt.status = MOI.OTHER_ERROR
            break
        end

        if opt.num_iters == opt.iteration_limit
            opt.verbose &&
                println("iteration limit ($(opt.num_iters)) reached; terminating")
            opt.status = :ITERATION_LIMIT
            break
        end

        if time() - opt.solve_time >= opt.time_limit
            opt.verbose && println("time limit ($(opt.time_limit)) reached; terminating")
            opt.status = :TIME_LIMIT
            break
        end

        is_cut_off = any(add_sep_cuts(k, opt) for k in eachindex(opt.con_sets))
        if !is_cut_off
            if opt.verbose
                println("no cuts were added; terminating")
            end
            opt.status = MOI.OPTIMAL
            break
        end

        opt.num_iters += 1
        flush(stdout)
    end
    return
end

function oa_solver_driven_method(opt::Optimizer)
    function lazy_callback(cb)
        opt.num_callbacks += 1
        is_cut_off = any(add_sep_cuts(k, opt, cb) for k in eachindex(opt.con_sets))
        if !is_cut_off && opt.verbose
            println("no cuts were added during callback")
        end
    end
    MOI.set(opt.oa_opt, MOI.LazyConstraintCallback(), lazy_callback)

    MOI.optimize!(opt.oa_opt)
    return
end
