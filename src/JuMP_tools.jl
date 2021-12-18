# tools for JuMP functions

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
