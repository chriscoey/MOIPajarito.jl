# conic and OA models for OA algorithms

# setup conic and OA models
function setup_models(opt::Optimizer)
    has_eq = !isempty(opt.b)

    # mixed-integer OA model
    oa_model = opt.oa_model = JuMP.Model(() -> opt.oa_opt)
    oa_x = opt.oa_x = JuMP.@variable(oa_model, [1:length(opt.c)])
    for i in 1:(opt.num_int_vars)
        JuMP.set_integer(oa_x[i])
    end
    JuMP.@objective(oa_model, Min, JuMP.dot(opt.c, oa_x))
    if has_eq
        JuMP.@constraint(oa_model, opt.A * oa_x .== opt.b)
    end

    # continuous relaxation model
    relax_model = opt.relax_model = JuMP.Model(() -> opt.conic_opt)
    relax_x = opt.relax_x = JuMP.@variable(relax_model, [1:length(opt.c)])
    JuMP.@objective(relax_model, Min, JuMP.dot(opt.c, relax_x))
    if has_eq
        JuMP.@constraint(relax_model, opt.A * relax_x .== opt.b)
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
    subp_x = opt.subp_x = JuMP.@variable(subp_model, [1:num_cont_vars])
    JuMP.@objective(subp_model, Min, JuMP.dot(c_cont, subp_x))
    opt.subp_eq = if has_eq
        JuMP.@constraint(subp_model, -A_cont * subp_x in MOI.Zeros(length(opt.b_cont)))
    else
        nothing
    end

    # conic constraints
    oa_aff = JuMP.@expression(oa_model, opt.h - opt.G * oa_x)
    relax_aff = JuMP.@expression(relax_model, opt.h - opt.G * relax_x)
    subp_aff = JuMP.@expression(subp_model, -G_cont * subp_x)

    oa_vars = opt.oa_vars = copy(oa_x)
    opt.subp_cones = CR[]
    opt.subp_cone_idxs = UnitRange{Int}[]
    opt.relax_oa_cones = CR[]
    opt.subp_oa_cones = CR[]
    opt.cone_caches = Cones.ConeCache[]
    opt.oa_cone_idxs = UnitRange{Int}[]
    opt.oa_slack_idxs = UnitRange{Int}[]

    for (cone, idxs) in zip(opt.cones, opt.cone_idxs)
        relax_cone_i = JuMP.@constraint(relax_model, relax_aff[idxs] in cone)

        oa_supports = MOI.supports_constraint(opt.oa_opt, VAF, typeof(cone))

        subp_aff_i = subp_aff[idxs]
        if !oa_supports || !iszero(subp_aff_i)
            # conic constraint must be added to subproblem
            subp_cone_i = JuMP.@constraint(subp_model, subp_aff_i in cone)
            push!(opt.subp_cones, subp_cone_i)
            push!(opt.subp_cone_idxs, idxs)
        end

        oa_aff_i = oa_aff[idxs]
        if oa_supports
            JuMP.@constraint(oa_model, oa_aff_i in cone)
        else
            # add slack variables where useful and modify oa_aff_i
            (slacks, slack_idxs) = create_slacks(oa_model, oa_aff_i)
            append!(oa_vars, slacks)
            push!(opt.oa_slack_idxs, idxs[slack_idxs])

            # set up cone cache and extended formulation
            cache = Cones.create_cache(oa_aff_i, cone, opt.use_extended_form)
            ext_i = Cones.setup_auxiliary(cache, oa_model)
            append!(oa_vars, ext_i)

            push!(opt.relax_oa_cones, relax_cone_i)
            push!(opt.subp_oa_cones, subp_cone_i)
            push!(opt.cone_caches, cache)
            push!(opt.oa_cone_idxs, idxs)
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

    isempty(opt.cone_caches) || return false
    if opt.verbose
        println("no conic constraints need outer approximation")
    end

    opt.debug_cuts && return false

    # no conic constraints need outer approximation, so just solve the OA model and finish
    time_finish = check_set_time_limit(opt, oa_model)
    time_finish && return true
    JuMP.optimize!(oa_model)

    opt.status = JuMP.termination_status(oa_model)
    if opt.status == MOI.OPTIMAL
        opt.obj_value = JuMP.objective_value(oa_model)
        opt.obj_bound = get_objective_bound(oa_model)
        opt.incumbent = JuMP.value.(oa_x)
    end
    return true
end

function create_slacks(model::JuMP.Model, expr_vec::Vector{AE})
    slacks = VR[]
    slack_idxs = Int[]
    for (j, expr_j) in enumerate(expr_vec)
        terms = JuMP.linear_terms(expr_j)
        length(terms) <= 1 && continue

        # affine expression has more than one variable, so add a slack variable
        s_j = JuMP.@variable(model)
        JuMP.@constraint(model, s_j .== expr_j)
        expr_vec[j] = s_j
        push!(slacks, s_j)
        push!(slack_idxs, j)
    end
    return (slacks, slack_idxs)
end

function get_oa_start(opt::Optimizer, x_start::Vector{Float64})
    n = length(opt.incumbent)
    @assert length(x_start) == n
    oa_start = fill(NaN, length(opt.oa_vars))
    oa_start[1:n] .= x_start

    s_start = opt.h - opt.G * x_start
    for (i, cache) in enumerate(opt.cone_caches)
        slack_idxs = opt.oa_slack_idxs[i]
        if !isempty(slack_idxs)
            # set slack variables start
            slack_start = s_start[slack_idxs]
            dim = length(slack_start)
            oa_start[n .+ (1:dim)] .= slack_start
            n += dim
        end

        ext_dim = Cones.num_ext_variables(cache)
        if !iszero(ext_dim)
            # set auxiliary variables start
            s_start_i = s_start[opt.oa_cone_idxs[i]]
            ext_start = Cones.extend_start(cache, s_start_i)
            @assert ext_dim == length(ext_start)
            oa_start[n .+ (1:ext_dim)] .= ext_start
            n += ext_dim
        end
    end
    @assert n == length(oa_start)

    @assert !any(isnan, oa_start)
    return oa_start
end

function modify_subproblem(int_sol::Vector{Int}, opt::Optimizer)
    # TODO maybe also modify the objective constant using dot(opt.c_int, int_sol), could be nonzero
    moi_model = JuMP.backend(opt.subp_model)

    if !isnothing(opt.subp_eq)
        new_b = opt.b_cont - opt.A_int * int_sol
        MOI.modify(moi_model, JuMP.index(opt.subp_eq), MOI.VectorConstantChange(new_b))
    end

    new_h = opt.h - opt.G_int * int_sol
    for (cr, idxs) in zip(opt.subp_cones, opt.subp_cone_idxs)
        MOI.modify(moi_model, JuMP.index(cr), MOI.VectorConstantChange(new_h[idxs]))
    end
    return
end
