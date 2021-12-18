# conic and OA models for OA algorithms

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
    oa_vars = opt.oa_vars = copy(x_oa)
    opt.subp_cones = CR[]
    opt.subp_cone_idxs = UnitRange{Int}[]
    opt.relax_oa_cones = CR[]
    opt.subp_oa_cones = CR[]
    opt.cone_caches = Cones.ConeCache[]
    opt.oa_cone_idxs = UnitRange{Int}[]

    for (cone, idxs) in zip(opt.cones, opt.cone_idxs)
        # TODO should keep h_i and G_i separate for the individual cone constraints
        h_i = opt.h[idxs]
        G_i = opt.G[idxs, :]
        G_cont_i = G_cont[idxs, :]

        relax_cone_i = JuMP.@constraint(relax_model, h_i - G_i * x_relax in cone)

        oa_supports = MOI.supports_constraint(opt.oa_opt, VAF, typeof(cone))
        if !oa_supports || !iszero(G_cont_i)
            subp_cone_i = JuMP.@constraint(subp_model, -G_cont_i * x_subp in cone)
            push!(opt.subp_cones, subp_cone_i)
            push!(opt.subp_cone_idxs, idxs)
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

function get_oa_start(opt::Optimizer, x_start::Vector{Float64})
    n = length(opt.incumbent)
    @assert length(x_start) == n

    oa_start = fill(NaN, length(opt.oa_vars))
    fill!(oa_start, NaN)
    oa_start[1:n] .= x_start

    s_start = opt.h - opt.G * x_start
    for (cache, idxs) in zip(opt.cone_caches, opt.oa_cone_idxs)
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
