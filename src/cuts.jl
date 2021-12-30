# helpers for cuts

# add initial fixed cuts
function add_init_cuts(opt::Optimizer)
    for cache in opt.cone_caches
        opt.num_cuts += Cones.add_init_cuts(cache, opt.oa_model)
    end
    return
end

# add relaxation dual cuts
function add_relax_cuts(opt::Optimizer)
    JuMP.has_duals(opt.relax_model) || return 0

    num_cuts_before = opt.num_cuts
    for (ci, cache) in zip(opt.relax_oa_cones, opt.cone_caches)
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
function add_subp_cuts(opt::Optimizer, cuts_cache::Union{Nothing, Vector{AE}} = nothing)
    JuMP.has_duals(opt.subp_model) || return false

    num_cuts_before = opt.num_cuts
    for (ci, cache) in zip(opt.subp_oa_cones, opt.cone_caches)
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

# add separation cuts
function add_sep_cuts(opt::Optimizer)
    num_cuts_before = opt.num_cuts

    s_vals = [get_value(Cones.get_oa_s(cache), opt.lazy_cb) for cache in opt.cone_caches]
    for (s, cache) in zip(s_vals, opt.cone_caches)
        cuts = Cones.get_sep_cuts(s, cache, opt.oa_model)
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
