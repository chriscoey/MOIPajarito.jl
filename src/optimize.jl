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
    use_extended_form::Bool
    oa_solver::Union{Nothing, MOI.OptimizerWithAttributes}
    conic_solver::Union{Nothing, MOI.OptimizerWithAttributes}

    # optimizers
    oa_opt::Union{Nothing, MOI.AbstractOptimizer}
    conic_opt::Union{Nothing, MOI.AbstractOptimizer}

    # problem data
    obj_sense::MOI.OptimizationSense
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
    x_oa::Vector{VR}
    x_relax::Vector{VR}
    x_subp::Vector{VR}
    subp_eq::Union{Nothing, CR}
    subp_cones::Vector{Tuple{CR, UnitRange{Int}}}
    c_int::Vector{Float64}
    A_int::SparseArrays.SparseMatrixCSC{Float64, Int}
    G_int::SparseArrays.SparseMatrixCSC{Float64, Int}
    b_cont::Vector{Float64}
    oa_cones::Vector{Tuple{CR, CR, Cones.ConeCache, Vector{VR}, UnitRange{Int}}}
    oa_vars::Vector{VR}

    # used/modified during optimize
    lazy_cb::Any
    new_incumbent::Bool
    int_sols_cuts::Dict{UInt, Vector{JuMP.AffExpr}}
    use_oa_starts::Bool

    # used by MOI wrapper
    incumbent::Vector{Float64}
    warm_start::Vector{Float64}
    status::MOI.TerminationStatusCode
    solve_time::Float64
    obj_value::Float64
    obj_bound::Float64
    num_cuts::Int
    num_iters::Int
    num_lazy_cbs::Int
    num_heuristic_cbs::Int

    function Optimizer(
        verbose::Bool = true,
        tol_feas::Float64 = 1e-7,
        tol_rel_gap::Float64 = 1e-5,
        tol_abs_gap::Float64 = 1e-4,
        time_limit::Float64 = 1e6,
        iteration_limit::Int = 1000,
        use_iterative_method::Union{Nothing, Bool} = nothing,
        use_extended_form::Bool = true,
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
        opt.use_extended_form = use_extended_form
        opt.oa_solver = oa_solver
        opt.conic_solver = conic_solver
        opt.oa_opt = nothing
        opt.conic_opt = nothing
        return empty_optimize(opt)
    end
end

function empty_optimize(opt::Optimizer)
    opt.status = MOI.OPTIMIZE_NOT_CALLED
    opt.solve_time = NaN
    opt.obj_value = NaN
    opt.obj_bound = NaN
    opt.num_cuts = 0
    opt.num_iters = 0
    opt.num_lazy_cbs = 0
    opt.num_heuristic_cbs = 0
    opt.lazy_cb = nothing
    opt.new_incumbent = false
    opt.int_sols_cuts = Dict{UInt, Vector{JuMP.AffExpr}}()
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
    # solve OA model
    time_finish = check_set_time_limit(opt, opt.oa_model)
    time_finish && return true
    JuMP.optimize!(opt.oa_model)

    oa_status = JuMP.termination_status(opt.oa_model)
    if oa_status == MOI.INFEASIBLE
        if opt.verbose
            println("infeasibility detected while iterating; terminating")
        end
        opt.status = oa_status
        return true
    elseif oa_status == MOI.TIME_LIMIT
        if opt.verbose
            println("OA solver timed out")
        end
        opt.status = oa_status
        return true
    elseif oa_status != MOI.OPTIMAL
        @warn("OA solver status $oa_status is not handled")
        opt.status = MOI.OTHER_ERROR
        return true
    end

    # update objective bound
    opt.obj_bound = get_objective_bound(opt.oa_model)

    # solve conic subproblem with fixed integer solution and update incumbent
    # check if integer solution is repeated
    subp_cuts_added = false
    int_sol = get_integral_solution(opt)
    hash_int_sol = hash(int_sol)
    if haskey(opt.int_sols_cuts, hash_int_sol)
        @warn("integral solution repeated")
    else
        # new integral solution: solve subproblem and add cuts
        opt.int_sols_cuts[hash_int_sol] = JuMP.AffExpr[]
        subp_failed = solve_subproblem(int_sol, opt)
        if !subp_failed
            subp_cuts_added = add_subp_cuts(opt)
        end
    end

    if !subp_cuts_added
        # add separation cuts and update incumbent from OA solution if no cuts are added
        # TODO should check feas whether or not cuts are added, and accept incumbent if feas
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

    # set warm start from incumbent
    if opt.use_oa_starts && isfinite(opt.obj_value)
        oa_start = get_oa_start(opt, opt.incumbent)
        JuMP.set_start_value.(opt.oa_vars, oa_start)
    end
    return false
end

# one tree method using callbacks
function run_one_tree_method(opt::Optimizer)
    if opt.verbose
        println("starting one tree method")
    end
    oa_model = opt.oa_model

    function lazy_cb(cb)
        opt.num_lazy_cbs += 1
        cb_status = JuMP.callback_node_status(cb, oa_model)
        if cb_status != MOI.CALLBACK_NODE_STATUS_INTEGER
            # only solve subproblem at an integer solution
            return
        end
        opt.lazy_cb = cb

        # check if integer solution is repeated and cache the cuts
        subp_cuts_added = false
        int_sol = get_integral_solution(opt)
        hash_int_sol = hash(int_sol)
        if haskey(opt.int_sols_cuts, hash_int_sol)
            # integral solution repeated: add cached cuts
            cuts = opt.int_sols_cuts[hash_int_sol]
            num_cuts_before = opt.num_cuts
            add_cuts(cuts, opt)
            if opt.num_cuts <= num_cuts_before
                if opt.verbose
                    println("cached subproblem cuts could not be added")
                end
            else
                subp_cuts_added = true
            end
        else
            # new integral solution: solve subproblem, cache cuts, and add cuts
            subp_failed = solve_subproblem(int_sol, opt)
            cuts_cache = opt.int_sols_cuts[hash_int_sol] = JuMP.AffExpr[]
            if !subp_failed
                subp_cuts_added = add_subp_cuts(opt, cuts_cache)
            end
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
    MOI.set(oa_model, MOI.LazyConstraintCallback(), lazy_cb)

    function heuristic_cb(cb)
        opt.num_heuristic_cbs += 1
        opt.new_incumbent || return true
        oa_start = get_oa_start(opt, opt.incumbent)
        cb_status = MOI.submit(oa_model, MOI.HeuristicSolution(cb), opt.oa_vars, oa_start)
        println("heuristic cb status was: ", cb_status)
        # TODO do what with cb_status
        opt.new_incumbent = false
        return
    end
    MOI.set(oa_model, MOI.HeuristicCallback(), heuristic_cb)

    time_finish = check_set_time_limit(opt, oa_model)
    time_finish && return true
    JuMP.optimize!(oa_model)

    oa_status = JuMP.termination_status(oa_model)
    if oa_status == MOI.OPTIMAL
        opt.status = oa_status
        opt.obj_bound = get_objective_bound(oa_model)
    elseif oa_status == MOI.INFEASIBLE
        opt.status = oa_status
    elseif oa_status == MOI.TIME_LIMIT
        if opt.verbose
            println("OA solver timed out")
        end
        opt.status = oa_status
    else
        @warn("OA solver status $oa_status is not handled")
        opt.status = MOI.OTHER_ERROR
    end
    return
end

function solve_relaxation(opt::Optimizer)
    if opt.verbose
        println("solving continuous relaxation")
    end
    time_finish = check_set_time_limit(opt, opt.relax_model)
    time_finish && return true
    JuMP.optimize!(opt.relax_model)

    relax_status = JuMP.termination_status(opt.relax_model)
    if opt.verbose
        println("continuous relaxation status is $relax_status")
    end
    if relax_status in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
        opt.obj_bound = JuMP.dual_objective_value(opt.relax_model)
    elseif relax_status in (MOI.DUAL_INFEASIBLE, MOI.ALMOST_DUAL_INFEASIBLE)
        if opt.verbose
            println("problem could be unbounded; Pajarito may fail to converge")
        end
    elseif relax_status in (MOI.INFEASIBLE, MOI.ALMOST_INFEASIBLE)
        if opt.verbose
            println("infeasibility detected from continuous relaxation; terminating")
        end
        opt.status = MOI.INFEASIBLE
        return true
    elseif relax_status == MOI.TIME_LIMIT
        if opt.verbose
            println("continuous relaxation solver timed out")
        end
        opt.status = relax_status
        return true
    else
        @warn("continuous relaxation status $relax_status is not handled")
        return false
    end

    finish = false

    # check whether problem is continuous
    if iszero(opt.num_int_vars)
        if opt.verbose
            println("problem is continuous; terminating")
        end
        finish = true
    end

    if relax_status in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
        # check whether conic relaxation solution is integral
        int_sol = JuMP.value.(opt.x_relax[1:(opt.num_int_vars)])
        round_int_sol = round.(Int, int_sol)
        # TODO different tol option?
        if isapprox(round_int_sol, int_sol, atol = opt.tol_feas, rtol = opt.tol_feas)
            if opt.verbose
                println("optimal solution to conic relaxation is integral; terminating")
            end
            finish = true
        end
    end

    if finish
        if relax_status in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
            opt.status = MOI.OPTIMAL
            opt.obj_value = JuMP.objective_value(opt.relax_model)
            opt.incumbent = JuMP.value.(opt.x_relax)
        elseif relax_status in (MOI.DUAL_INFEASIBLE, MOI.ALMOST_DUAL_INFEASIBLE)
            opt.status = MOI.DUAL_INFEASIBLE
        else
            error("status $(opt.status) not handled")
        end
        return true
    end
    return false
end

# solve subproblem with new integer variable sub-solution
function solve_subproblem(int_sol::Vector{Int}, opt::Optimizer)
    # TODO maybe also modify the objective constant using dot(opt.c_int, int_sol), could be nonzero
    moi_model = JuMP.backend(opt.subp_model)
    if !isnothing(opt.subp_eq)
        new_b = opt.b_cont - opt.A_int * int_sol
        MOI.modify(moi_model, JuMP.index(opt.subp_eq), MOI.VectorConstantChange(new_b))
    end
    new_h = opt.h - opt.G_int * int_sol
    for (cr, idxs) in opt.subp_cones
        MOI.modify(moi_model, JuMP.index(cr), MOI.VectorConstantChange(new_h[idxs]))
    end

    # solve
    time_finish = check_set_time_limit(opt, opt.subp_model)
    time_finish && return true
    JuMP.optimize!(opt.subp_model)

    subp_status = JuMP.termination_status(opt.subp_model)
    if opt.verbose
        println("continuous subproblem status is $subp_status")
    end
    if subp_status in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
        obj_val =
            JuMP.objective_value(opt.subp_model) + LinearAlgebra.dot(opt.c_int, int_sol)
        if obj_val < opt.obj_value
            # update incumbent and objective value
            subp_sol = JuMP.value.(opt.x_subp)
            opt.incumbent = vcat(int_sol, subp_sol)
            opt.obj_value = obj_val
            # println("new incumbent")
            opt.new_incumbent = true
        end
        return false
    elseif subp_status in (MOI.INFEASIBLE, MOI.ALMOST_INFEASIBLE)
        # NOTE: duals are rescaled before adding subproblem cuts
        return false
    elseif subp_status == MOI.TIME_LIMIT
        if opt.verbose
            println("continuous subproblem solver timed out")
        end
        opt.status = subp_status
        return true
    else
        @warn("continuous subproblem status $subp_status is not handled")
        return false
    end
end

# setup conic and OA models
function setup_models(opt::Optimizer)
    has_eq = !isempty(opt.b)

    # mixed-integer OA model
    oa_model = opt.oa_model = JuMP.Model(() -> opt.oa_opt)
    x_oa = opt.x_oa = JuMP.@variable(oa_model, [1:length(opt.c)])
    for i in 1:(opt.num_int_vars)
        JuMP.set_integer(x_oa[i])
    end
    JuMP.@objective(oa_model, Min, JuMP.dot(opt.c, x_oa))
    if has_eq
        JuMP.@constraint(oa_model, opt.A * x_oa .== opt.b)
    end

    # continuous relaxation model
    relax_model = opt.relax_model = JuMP.Model(() -> opt.conic_opt)
    x_relax = opt.x_relax = JuMP.@variable(relax_model, [1:length(opt.c)])
    JuMP.@objective(relax_model, Min, JuMP.dot(opt.c, x_relax))
    if has_eq
        JuMP.@constraint(relax_model, opt.A * x_relax .== opt.b)
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
        A_cont = opt.A[:, cont_range]
        # remove zero rows in A_cont for subproblem
        keep_rows = (vec(maximum(abs, A_cont, dims = 2)) .>= 1e-10)
        A_cont = A_cont[keep_rows, :]
        opt.b_cont = opt.b[keep_rows]
        opt.A_int = opt.A[keep_rows, int_range]
    else
        opt.b_cont = Float64[]
        opt.A_int = zeros(Float64, 0, opt.num_int_vars)
    end

    # continuous subproblem model
    subp_model = opt.subp_model = JuMP.Model(() -> opt.conic_opt)
    x_subp = opt.x_subp = JuMP.@variable(subp_model, [1:num_cont_vars])
    JuMP.@objective(subp_model, Min, JuMP.dot(c_cont, x_subp))
    opt.subp_eq = if has_eq
        JuMP.@constraint(subp_model, -A_cont * x_subp in MOI.Zeros(length(opt.b_cont)))
    else
        nothing
    end

    # conic constraints
    opt.oa_cones = Tuple{CR, CR, Cones.ConeCache, Vector{VR}, UnitRange{Int}}[]
    opt.subp_cones = Tuple{CR, UnitRange{Int}}[]
    oa_vars = opt.oa_vars = copy(x_oa)

    for (cone, idxs) in zip(opt.cones, opt.cone_idxs)
        # TODO should keep h_i and G_i separate for the individual cone constraints
        h_i = opt.h[idxs]
        G_i = opt.G[idxs, :]
        G_cont_i = G_cont[idxs, :]

        K_relax_i = JuMP.@constraint(relax_model, h_i - G_i * x_relax in cone)

        oa_supports = MOI.supports_constraint(opt.oa_opt, VAF, typeof(cone))
        if !oa_supports || !iszero(G_cont_i)
            K_subp_i = JuMP.@constraint(subp_model, -G_cont_i * x_subp in cone)
            push!(opt.subp_cones, (K_subp_i, idxs))
        end

        if oa_supports
            JuMP.@constraint(oa_model, h_i - G_i * x_oa in cone)
        else
            # TODO don't add slacks if h_i = 0 and G_i = -I (i.e. original constraint was a VV)
            s_i = JuMP.@variable(oa_model, [1:length(idxs)])
            append!(oa_vars, s_i)
            JuMP.@constraint(oa_model, s_i .== h_i - G_i * x_oa)

            # set up cone cache and extended formulation
            cache = Cones.create_cache(s_i, cone, opt.use_extended_form)
            ext_i = Cones.setup_auxiliary(cache, oa_model)
            append!(oa_vars, ext_i)

            push!(opt.oa_cones, (K_relax_i, K_subp_i, cache, s_i, idxs))
        end
    end
    @assert JuMP.num_variables(oa_model) == length(oa_vars)

    opt.use_oa_starts = MOI.supports(JuMP.backend(oa_model), MOI.VariablePrimalStart(), VI)
    if opt.use_oa_starts && !isempty(opt.warm_start)
        if any(isnan, opt.warm_start)
            @warn("warm start is only partial so will be ignored")
        else
            oa_start = get_oa_start(opt, opt.warm_start)
            JuMP.set_start_value.(oa_vars, oa_start)
        end
    end

    isempty(opt.oa_cones) || return false
    if opt.verbose
        println("no conic constraints need outer approximation")
    end

    # no conic constraints need outer approximation, so just solve the OA model and finish
    time_finish = check_set_time_limit(opt, oa_model)
    time_finish && return true
    JuMP.optimize!(oa_model)

    opt.status = JuMP.termination_status(oa_model)
    if opt.status == MOI.OPTIMAL
        opt.obj_value = JuMP.objective_value(oa_model)
        opt.obj_bound = get_objective_bound(oa_model)
        opt.incumbent = JuMP.value.(x_oa)
    end
    return true
end

# initial fixed cuts
function add_init_cuts(opt::Optimizer)
    for (_, _, cache, _, _) in opt.oa_cones
        opt.num_cuts += Cones.add_init_cuts(cache, opt.oa_model)
    end
    return
end

# add relaxation dual cuts
function add_relax_cuts(opt::Optimizer)
    JuMP.has_duals(opt.relax_model) || return 0

    num_cuts_before = opt.num_cuts
    for (ci, _, cache, _, _) in opt.oa_cones
        z = JuMP.dual(ci)
        z_norm = LinearAlgebra.norm(z, Inf)
        if z_norm < 1e-10 # TODO tune
            continue # discard duals with small norm
        elseif z_norm > 1e12
            @warn("norm of dual is large ($z_norm)")
        end

        cuts = Cones.get_subp_cuts(z, cache, opt.oa_model)
        add_cuts(cuts, opt)
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
function add_subp_cuts(
    opt::Optimizer,
    cuts_cache::Union{Nothing, Vector{JuMP.AffExpr}} = nothing,
)
    JuMP.has_duals(opt.subp_model) || return false

    num_cuts_before = opt.num_cuts
    for (_, ci, cache, _, _) in opt.oa_cones
        z = JuMP.dual(ci)
        z_norm = LinearAlgebra.norm(z, Inf)
        if z_norm < 1e-10 # TODO tune
            continue # discard duals with small norm
        elseif z_norm > 1e12
            @warn("norm of dual is large ($z_norm)")
        end
        subp_status = JuMP.termination_status(opt.subp_model)
        if subp_status in (MOI.INFEASIBLE, MOI.ALMOST_INFEASIBLE)
            # rescale dual rays
            z .*= inv(z_norm)
        end

        cuts = Cones.get_subp_cuts(z, cache, opt.oa_model)
        add_cuts(cuts, opt)

        if !isnothing(cuts_cache)
            append!(cuts_cache, cuts)
        end
    end
    return opt.num_cuts > num_cuts_before
end

# separation cuts
function add_sep_cuts(opt::Optimizer)
    num_cuts_before = opt.num_cuts
    for (_, _, cache, _, _) in opt.oa_cones
        Cones.load_s(cache, opt.lazy_cb)
    end
    for (_, _, cache, _, _) in opt.oa_cones
        cuts = Cones.get_sep_cuts(cache, opt.oa_model)
        add_cuts(cuts, opt)
    end

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
    sol = get_value(opt.x_oa, opt.lazy_cb)
    obj_val = LinearAlgebra.dot(opt.c, sol)
    if obj_val < opt.obj_value
        # update incumbent and objective value
        opt.incumbent = sol
        opt.obj_value = obj_val
        println("new incumbent")
        opt.new_incumbent = true
    end
    return
end

function get_integral_solution(opt::Optimizer)
    x_int = opt.x_oa[1:(opt.num_int_vars)]
    int_sol = get_value.(x_int, opt.lazy_cb)

    # check solution is integral
    round_int_sol = round.(Int, int_sol)
    # TODO different tol option?
    if !isapprox(round_int_sol, int_sol, atol = opt.tol_feas, rtol = opt.tol_feas)
        error("integer variable solution is not integral to tolerance tol_feas")
    end
    return round_int_sol
end

function get_oa_start(opt::Optimizer, x_start::Vector{Float64})
    n = length(opt.incumbent)
    @assert length(x_start) == n

    oa_start = fill(NaN, length(opt.oa_vars))
    fill!(oa_start, NaN)
    oa_start[1:n] .= x_start

    s_start = opt.h - opt.G * x_start
    for (_, _, cache, _, idxs) in opt.oa_cones
        s_start_i = s_start[idxs]
        dim = length(s_start_i)
        oa_start[n .+ (1:dim)] .= s_start_i
        n += dim
        ext_start = Cones.extend_start(cache, s_start_i)
        isempty(ext_start) && continue
        ext_dim = length(ext_start)
        oa_start[n .+ (1:ext_dim)] .= ext_start
        n += ext_dim
    end
    @assert n == length(oa_start)
    @assert !any(isnan, oa_start)
    return oa_start
end

# initialize and print
function start_optimize(opt::Optimizer)
    get_conic_opt(opt)
    get_oa_opt(opt)
    empty_optimize(opt)
    opt.solve_time = time()
    opt.obj_value = Inf
    opt.obj_bound = -Inf
    return
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
    return
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

# try to get an objective bound or dual objective value, else use the primal objective value
function get_objective_bound(model::JuMP.Model)
    try
        return JuMP.objective_bound(model)
    catch
    end
    try
        return JuMP.dual_objective_value(model)
    catch
    end
    return JuMP.objective_value(model)
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

function add_cuts(cuts::Vector{JuMP.AffExpr}, opt::Optimizer)
    for cut in cuts
        opt.num_cuts += add_cut(cut, opt)
    end
    return
end

function add_cut(cut::JuMP.AffExpr, opt::Optimizer)
    return _add_cut(cut, opt.oa_model, opt.tol_feas, opt.lazy_cb)
end

function _add_cut(cut::JuMP.AffExpr, model::JuMP.Model, ::Float64, ::Nothing)
    JuMP.@constraint(model, cut >= 0)
    return 1
end

function _add_cut(cut::JuMP.AffExpr, model::JuMP.Model, tol_feas::Float64, cb)
    # only add cut if violated (per JuMP documentation)
    if JuMP.callback_value(cb, cut) < -tol_feas
        con = JuMP.@build_constraint(cut >= 0)
        MOI.submit(model, MOI.LazyConstraint(cb), con)
        return 1
    end
    return 0
end

function check_set_time_limit(opt::Optimizer, model::JuMP.Model)
    time_left = opt.time_limit - time() + opt.solve_time
    if time_left < 1e-3
        if opt.verbose
            println("time limit ($(opt.time_limit)) reached; terminating")
        end
        opt.status = MOI.TIME_LIMIT
        return true
    end
    if MOI.supports(JuMP.backend(model), MOI.TimeLimitSec())
        JuMP.set_time_limit_sec(model, time_left)
    end
    return false
end
