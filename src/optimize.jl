# outer approximation algorithm

mutable struct Optimizer <: MOI.AbstractOptimizer
    # options
    verbose::Bool
    tol_feas::Float64
    tol_rel_gap::Float64
    tol_abs_gap::Float64
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
    # zeros_idxs::Vector{UnitRange{Int}} # TODO needed?

    # problem data
    obj_offset::Float64
    c::Vector{Float64}
    A::SparseArrays.SparseMatrixCSC{Float64, Int}
    b::Vector{Float64}
    G::SparseArrays.SparseMatrixCSC{Float64, Int}
    h::Vector{Float64}
    cones::Vector{AVS}
    cone_idxs::Vector{UnitRange{Int}}
    num_int_vars::Int

    # models, variables, etc
    oa_model::JuMP.Model
    relax_model::JuMP.Model
    subp_model::JuMP.Model
    oa_vars::Vector{VR}
    relax_vars::Vector{VR}
    subp_vars::Vector{VR}
    subp_eq::Union{Nothing, CR}
    subp_cones::Vector{CR}
    c_int::Vector{Float64}
    A_int::SparseArrays.SparseMatrixCSC{Float64, Int}
    G_int::SparseArrays.SparseMatrixCSC{Float64, Int}
    oa_cones::Vector{Tuple{CR, CR, Vector{VR}, AVS}}

    # modified throughout optimize and used after optimize
    status::MOI.TerminationStatusCode
    incumbent::Vector{Float64}
    obj_value::Float64
    obj_bound::Float64
    num_cuts::Int
    num_iters::Int
    lazy_cb::Any
    num_lazy_cbs::Int
    num_heuristic_cbs::Int
    new_incumbent::Bool
    solve_time::Float64

    function Optimizer(
        verbose::Bool = true,
        tol_feas::Float64 = 1e-7,
        tol_rel_gap::Float64 = 1e-5,
        tol_abs_gap::Float64 = 1e-7,
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
        opt.tol_abs_gap = tol_abs_gap
        opt.time_limit = time_limit
        opt.iteration_limit = iteration_limit
        opt.use_iterative_method = use_iterative_method
        opt.oa_solver = oa_solver
        opt.conic_solver = conic_solver
        opt.oa_opt = nothing
        opt.conic_opt = nothing
        return empty_all(opt)
    end
end

function empty_all(opt::Optimizer)
    opt.relax_vars = VR[]
    opt.oa_vars = VR[]
    opt.subp_vars = VR[]
    opt.incumbent = Float64[]
    empty_optimize(opt)
    return opt
end

function empty_optimize(opt::Optimizer)
    opt.status = MOI.OPTIMIZE_NOT_CALLED
    opt.obj_value = NaN
    opt.obj_bound = NaN
    opt.num_cuts = 0
    opt.num_iters = 0
    opt.lazy_cb = nothing
    opt.num_lazy_cbs = 0
    opt.num_heuristic_cbs = 0
    opt.new_incumbent = false
    opt.solve_time = NaN
    if !isnothing(opt.oa_opt)
        MOI.empty!(opt.oa_opt)
    end
    if !isnothing(opt.conic_opt)
        MOI.empty!(opt.conic_opt)
    end
    return opt
end

function optimize(opt::Optimizer)
    start_optimize(opt)

    # setup models
    setup_finish = setup_models(opt)
    if setup_finish
        finish_optimize(opt)
        return
    end

    # solve continuous relaxation
    relax_finish = solve_relaxation(opt)
    if relax_finish
        finish_optimize(opt)
        return
    end

    # add continuous relaxation cuts and initial fixed cuts
    add_relax_cuts(opt)
    add_init_cuts(opt)

    if opt.use_iterative_method
        # solve using iterative method
        while true
            opt.num_iters += 1
            finish_iter = run_iterative_method(opt)
            finish_iter && break
        end
    else
        # solve using one tree method
        run_one_tree_method(opt)
    end

    finish_optimize(opt)
    return
end

# one iteration of the iterative method
function run_iterative_method(opt::Optimizer)
    time_finish = check_time_limit(opt)
    if time_finish
        return true
    end

    # solve OA model
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
    subp_failed = solve_subproblem(opt)
    # subp_failed && return true

    # add OA cuts and update incumbent from OA solution if no cuts are added
    subp_cuts_added = false
    if !subp_failed
        subp_cuts_added = add_subp_cuts(opt)
    end
    if !subp_cuts_added
        cuts_added = add_sep_cuts(opt)
        if !cuts_added
            # no separation cuts, so try updating incumbent from OA solver
            update_incumbent_from_OA(opt)
        end
    end

    # print and check convergence
    obj_rel_gap = get_obj_rel_gap(opt)
    if opt.verbose
        Printf.@printf(
            "%5d %8d %12.4e %12.4e %12.4e\n",
            opt.num_iters,
            opt.num_cuts,
            opt.obj_value,
            opt.obj_bound,
            obj_rel_gap,
        )
    end
    if !isnan(obj_rel_gap) && obj_rel_gap < opt.tol_rel_gap
        if opt.verbose
            println("objective relative gap $obj_rel_gap reached; terminating")
        end
        opt.status = MOI.OPTIMAL
        return true
    end
    obj_abs_gap = get_obj_abs_gap(opt)
    if !isnan(obj_abs_gap) && obj_abs_gap < opt.tol_abs_gap
        if opt.verbose
            println("objective absolute gap $obj_abs_gap reached; terminating")
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
    return false
end

# one tree method using callbacks
function run_one_tree_method(opt::Optimizer)
    if opt.verbose
        println("starting one tree method")
    end

    function lazy_cb(cb)
        opt.num_lazy_cbs += 1
        cb_status = JuMP.callback_node_status(cb, opt.oa_model)
        if cb_status != MOI.CALLBACK_NODE_STATUS_INTEGER
            # only solve subproblem at an integer solution
            return
        end
        opt.lazy_cb = cb

        # TODO check if integer solution is repeated and cache the cuts
        # (including those not added the first time because not violated)
        subp_failed = solve_subproblem(opt)

        # add OA cuts and update incumbent from OA solution if no cuts are added
        subp_cuts_added = false
        if !subp_failed
            subp_cuts_added = add_subp_cuts(opt)
        end
        if !subp_cuts_added
            cuts_added = add_sep_cuts(opt)
            if !cuts_added
                # no separation cuts, so try updating incumbent from OA solver
                update_incumbent_from_OA(opt)
            end
        end
        return
    end
    MOI.set(opt.oa_model, MOI.LazyConstraintCallback(), lazy_cb)

    function heuristic_cb(cb)
        opt.num_heuristic_cbs += 1
        if !opt.new_incumbent
            return
        end
        cb_status =
            MOI.submit(opt.oa_model, MOI.HeuristicSolution(cb), opt.oa_vars, opt.incumbent)
        println("heuristic cb status was: ", cb_status)
        # TODO do what with cb_status
        opt.new_incumbent = false
        return
    end
    MOI.set(opt.oa_model, MOI.HeuristicCallback(), heuristic_cb)

    check_time_limit(opt)
    JuMP.optimize!(opt.oa_model)
    oa_status = JuMP.termination_status(opt.oa_model)

    # TODO handle statuses properly
    if oa_status == MOI.OPTIMAL
        opt.status = oa_status
        opt.obj_bound = JuMP.objective_bound(opt.oa_model)
    else
        opt.status = oa_status
    end

    return
end

function check_time_limit(opt::Optimizer)
    time_left = opt.time_limit - time() + opt.solve_time
    if time_left < 1e-3
        if opt.verbose
            println("time limit ($(opt.time_limit)) reached; terminating")
        end
        opt.status = MOI.TIME_LIMIT
        return true
    end
    JuMP.set_time_limit_sec(opt.oa_model, time_left)
    return false
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
    elseif relax_status == MOI.DUAL_INFEASIBLE || relax_status == MOI.ALMOST_OPTIMAL
        if opt.verbose
            println("continuous relaxation status is $relax_status")
        end
    elseif relax_status == MOI.INFEASIBLE
        if opt.verbose
            println("infeasibility detected from continuous relaxation; terminating")
        end
        opt.status = relax_status
        return true
    else
        @warn("continuous relaxation status $relax_status is not handled")
        # opt.status = MOI.OTHER_ERROR
        return false
    end
    # @assert JuMP.has_duals(opt.relax_model)
    # TODO if optimal solution to conic relaxation is integral, don't need to do OA

    if iszero(opt.num_int_vars)
        # problem is continuous
        if opt.verbose
            println("problem is continuous; terminating without using OA solver")
        end
        if relax_status == MOI.OPTIMAL
            opt.status = relax_status
            opt.obj_value = JuMP.objective_value(opt.relax_model)
            opt.incumbent = JuMP.value.(opt.relax_vars)
        end
        return true
    end
    return false
end

# solve subproblem with new integer variable bounds
# TODO handle non-equal bounds in lazy callback
function solve_subproblem(opt::Optimizer)
    x_int = opt.oa_vars[1:(opt.num_int_vars)]
    int_sol = get_value.(x_int, opt.lazy_cb)

    # TODO want JuMP MultirowChange, see https://github.com/jump-dev/JuMP.jl/issues/1183
    moi_model = JuMP.backend(opt.subp_model)
    if !isnothing(opt.subp_eq)
        new_b = opt.b - opt.A_int * int_sol
        MOI.modify(moi_model, JuMP.index(opt.subp_eq), MOI.VectorConstantChange(new_b))
    end
    new_h = opt.h - opt.G_int * int_sol
    for (cr, _, idxs) in zip(opt.subp_cones, opt.cones, opt.cone_idxs)
        MOI.modify(moi_model, JuMP.index(cr), MOI.VectorConstantChange(new_h[idxs]))
    end

    # solve
    JuMP.optimize!(opt.subp_model)
    subp_status = JuMP.termination_status(opt.subp_model)
    if opt.verbose
        println("continuous subproblem status is $subp_status")
    end

    if subp_status == MOI.OPTIMAL
        obj_val =
            JuMP.objective_value(opt.subp_model) + LinearAlgebra.dot(opt.c_int, int_sol)
        if obj_val < opt.obj_value
            # update incumbent and objective value
            subp_sol = JuMP.value.(opt.subp_vars)
            opt.incumbent[1:(opt.num_int_vars)] = int_sol
            opt.incumbent[(opt.num_int_vars + 1):end] = subp_sol
            opt.obj_value = obj_val
            # println("new incumbent")
            opt.new_incumbent = true
        end
        return false
    elseif subp_status == MOI.INFEASIBLE
        # NOTE: duals are rescaled before adding subproblem cuts
        return false
    end

    @warn("continuous subproblem status $subp_status is not handled")
    opt.status = MOI.OTHER_ERROR
    return true
end

# initialize and print
function start_optimize(opt::Optimizer)
    get_conic_opt(opt)
    get_oa_opt(opt)
    empty_optimize(opt)
    opt.solve_time = time()
    opt.obj_value = Inf
    opt.obj_bound = -Inf
    return nothing
end

# finalize and print
function finish_optimize(opt::Optimizer)
    opt.solve_time = time() - opt.solve_time

    oa_status = MOI.get(opt.oa_opt, MOI.TerminationStatus())
    if opt.verbose && oa_status != MOI.OPTIMIZE_NOT_CALLED
        println(
            "OA solver finished with status $oa_status, after $(opt.solve_time) " *
            "seconds and $(opt.num_cuts) cuts",
        )
        if opt.use_iterative_method
            println("iterative method used $(opt.num_iters) iterations")
        else
            println(
                "one tree method used $(opt.num_lazy_cbs) lazy " *
                "callbacks and $(opt.num_heuristic_cbs) heuristic callbacks",
            )
        end
    end
    opt.verbose && println()

    return nothing
end

# compute objective absolute gap
function get_obj_abs_gap(opt::Optimizer)
    if opt.obj_sense == MOI.FEASIBILITY_SENSE
        return NaN
    end
    return opt.obj_value - opt.obj_bound
end

# compute objective relative gap
# TODO decide whether to use 1e-5 constant
get_obj_rel_gap(opt::Optimizer) = get_obj_abs_gap(opt) / (1e-5 + abs(opt.obj_value))

# setup conic and OA models
function setup_models(opt::Optimizer)
    has_eq = !isempty(opt.b)

    # mixed-integer OA model
    oa = opt.oa_model = JuMP.Model(() -> opt.oa_opt)
    x_oa = JuMP.@variable(oa, [1:length(opt.c)])
    opt.oa_vars = x_oa
    for i in 1:(opt.num_int_vars)
        JuMP.set_integer(x_oa[i])
    end
    JuMP.@objective(oa, Min, JuMP.dot(opt.c, x_oa))
    if has_eq
        JuMP.@constraint(oa, opt.A * x_oa .== opt.b)
    end

    # continuous relaxation model
    relax = opt.relax_model = JuMP.Model(() -> opt.conic_opt)
    x_relax = JuMP.@variable(relax, [1:length(opt.c)])
    opt.relax_vars = x_relax
    JuMP.@objective(relax, Min, JuMP.dot(opt.c, x_relax))
    if has_eq
        JuMP.@constraint(relax, opt.A * x_relax .== opt.b)
    end

    # differentiate integer and continuous variables
    num_cont_vars = length(opt.c) - opt.num_int_vars
    int_range = 1:(opt.num_int_vars)
    cont_range = (opt.num_int_vars + 1):length(opt.c)
    opt.c_int = opt.c[int_range]
    opt.G_int = opt.G[:, int_range]
    c_cont = opt.c[cont_range]
    G_cont = opt.G[:, cont_range]
    if has_eq
        opt.A_int = opt.A[:, int_range]
        A_cont = opt.A[:, cont_range]
    end

    # continuous subproblem model
    subp = opt.subp_model = JuMP.Model(() -> opt.conic_opt)
    x_subp = JuMP.@variable(subp, [1:num_cont_vars])
    opt.subp_vars = x_subp
    JuMP.@objective(subp, Min, JuMP.dot(c_cont, x_subp))
    if has_eq
        opt.subp_eq = JuMP.@constraint(subp, -A_cont * x_subp in MOI.Zeros(length(opt.b)))
    else
        opt.subp_eq = nothing
    end

    # conic constraints
    opt.oa_cones = Tuple{CR, CR, Vector{VR}, AVS}[]
    opt.subp_cones = CR[]
    for (cone, idxs) in zip(opt.cones, opt.cone_idxs)
        # TODO should keep h_i and G_i separate for the individual cone constraints
        h_i = opt.h[idxs]
        G_i = opt.G[idxs, :]
        G_cont_i = G_cont[idxs, :]

        K_relax_i = JuMP.@constraint(relax, h_i - G_i * x_relax in cone)
        K_subp_i = JuMP.@constraint(subp, -G_cont_i * x_subp in cone)
        push!(opt.subp_cones, K_subp_i)

        if MOI.supports_constraint(opt.oa_opt, VAF, typeof(cone))
            JuMP.@constraint(oa, h_i - G_i * x_oa in cone)
        else
            # TODO don't add slacks if h_i = 0 and G_i = -I (i.e. original constraint was a VV)
            s_i = JuMP.@variable(oa, [1:length(idxs)])
            JuMP.@constraint(oa, s_i .== h_i - G_i * x_oa)
            push!(opt.oa_cones, (K_relax_i, K_subp_i, s_i, cone))
        end
    end

    isempty(opt.oa_cones) || return false
    # no conic constraints need outer approximation, so just solve the OA model and finish
    if opt.verbose
        println("no conic constraints need outer approximation")
    end

    JuMP.optimize!(opt.oa_model)
    opt.status = JuMP.termination_status(opt.oa_model)
    if opt.status == MOI.OPTIMAL
        opt.obj_value = JuMP.objective_value(opt.oa_model)
        opt.obj_bound = JuMP.objective_bound(opt.oa_model)
        opt.incumbent = JuMP.value.(opt.oa_vars)
    end
    return true
end

function get_value(var::VR, ::Nothing)
    return JuMP.value(var)
end

function get_value(var::VR, cb)
    return JuMP.callback_value(cb, var)
end

function get_value(vars::Vector{VR}, ::Nothing)
    return JuMP.value.(vars)
end

function get_value(vars::Vector{VR}, cb)
    return JuMP.callback_value.(cb, vars)
end

# initial fixed cuts
function add_init_cuts(opt::Optimizer)
    for (_, _, s_vars, cone) in opt.oa_cones
        opt.num_cuts += Cuts.add_init_cuts(opt, s_vars, cone)
    end
    return nothing
end

# add relaxation dual cuts
function add_relax_cuts(opt::Optimizer)
    if !JuMP.has_duals(opt.relax_model)
        return 0
    end

    num_cuts_before = opt.num_cuts
    for (ci, _, s_vars, cone) in opt.oa_cones
        z = JuMP.dual(ci)
        z_norm = LinearAlgebra.norm(z, Inf)
        if z_norm < 1e-10 # TODO tune
            continue # discard duals with small norm
        elseif z_norm > 1e12
            @warn("norm of dual is large ($z_norm)")
        end
        opt.num_cuts += Cuts.add_subp_cuts(opt, z, s_vars, cone)
    end

    if opt.num_cuts <= num_cuts_before
        if opt.verbose
            println("continuous relaxation cuts could not be added")
        end
        return false
    end
    return true
end

# add subproblem dual cuts
# TODO save the list of cut expressions (all, even if not added this round)
# to add at repeated integral solutions.
function add_subp_cuts(opt::Optimizer)
    if !JuMP.has_duals(opt.subp_model)
        return 0
    end

    num_cuts_before = opt.num_cuts
    for (_, ci, s_vars, cone) in opt.oa_cones
        z = JuMP.dual(ci)
        z_norm = LinearAlgebra.norm(z, Inf)
        if z_norm < 1e-10 # TODO tune
            continue # discard duals with small norm
        elseif z_norm > 1e12
            @warn("norm of dual is large ($z_norm)")
        end
        if JuMP.termination_status(opt.subp_model) == MOI.INFEASIBLE
            # rescale dual rays
            z .*= inv(z_norm)
        end
        opt.num_cuts += Cuts.add_subp_cuts(opt, z, s_vars, cone)
    end

    if opt.num_cuts <= num_cuts_before
        if opt.verbose
            println("subproblem cuts could not be added")
        end
        return false
    end
    return true
end

# separation cuts
function add_sep_cuts(opt::Optimizer)
    num_cuts_before = opt.num_cuts
    # TODO can't get the values after added a cut, due to "optimize not called"
    ss = [get_value(s_vars, opt.lazy_cb) for (_, _, s_vars, _) in opt.oa_cones]
    for (s, (_, _, s_vars, cone)) in zip(ss, opt.oa_cones)
        opt.num_cuts += Cuts.add_sep_cuts(opt, s, s_vars, cone)
    end
    # for (_, _, s_vars, cone) in opt.oa_cones
    #     s = get_value(s_vars, opt.lazy_cb)
    #     opt.num_cuts += Cuts.add_sep_cuts(opt, s, s_vars, cone)
    # end

    if opt.num_cuts <= num_cuts_before
        if opt.verbose
            println("separation cuts could not be added")
        end
        return false
    end
    return true
end

# update incumbent from OA solver
function update_incumbent_from_OA(opt::Optimizer)
    # obj_val = JuMP.objective_value(opt.oa_model)
    sol = get_value(opt.oa_vars, opt.lazy_cb)
    obj_val = LinearAlgebra.dot(opt.c, sol)
    if obj_val < opt.obj_value
        # update incumbent and objective value
        copyto!(opt.incumbent, sol)
        opt.obj_value = obj_val
        # println("new incumbent")
        opt.new_incumbent = true
    end
end
