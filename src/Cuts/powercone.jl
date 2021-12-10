#=
power cone, parametrized by exponent t ∈ (0, 1)
(u, v, w) : u ≥ 0, v ≥ 0, u^t * v^(1-t) ≥ |w|
equivalently (v > 0, u ≥ v * |w / v|^(1/t))
dual cone
(p, q, r) : p ≥ 0, q ≥ 0, (p/t)^t * (q/(1-t))^(1-t) ≥ |r|
equivalently (q > 0, p ≥ t / (1-t) * q * |(1-t) * r / q|^(1/t))
=#

function add_init_cuts(opt::Optimizer, s_vars::Vector{VR}, cone::MOI.PowerCone)
    @assert length(s_vars) == 3
    (u, v, w) = s_vars
    t = cone.exponent
    # add variable bounds and cuts (t, 1-t, ±1)
    JuMP.@constraints(opt.oa_model, begin
        u >= 0
        v >= 0
        t * u + (1 - t) * v + w >= 0
        t * u + (1 - t) * v - w >= 0
    end)
    return 4
end

function add_subp_cuts(
    opt::Optimizer,
    z::Vector{Float64},
    s_vars::Vector{VR},
    cone::MOI.PowerCone,
)
    p = z[1]
    q = z[2]
    if min(p, q) <= 0
        # z ∉ K
        @warn("dual vector is not in the dual cone")
        return 0
    end

    # strengthened cut is (p, q, sign(r) * (p/t)^t * (q/(1-t))^(1-t))
    t = cone.exponent
    r = sign(z[3]) * (p / t)^t * (q / (1 - t))^(1 - t)
    (u, v, w) = s_vars
    expr = JuMP.@expression(opt.oa_model, p * u + q * v + r * w)
    return add_cut(expr, opt)
end

function add_sep_cuts(
    opt::Optimizer,
    s::Vector{Float64},
    s_vars::Vector{VR},
    cone::MOI.PowerCone,
)
    (us, vs, ws) = s
    if min(us, vs) <= -opt.tol_feas
        error("power cone point violates initial cuts")
    end

    # check s ∉ K
    t = cone.exponent
    if us >= 0 && vs >= 0 && (us^t * vs^(1 - t) - abs(ws)) > -opt.tol_feas
        return 0
    end

    # gradient cut is (t * (us/vs)^(t-1), (1-t) * (us/vs)^t, -sign(ws))
    # TODO need better approach when u or v near zero
    us = max(us, 1e-8)
    vs = max(vs, 1e-8)
    (u, v, w) = s_vars
    p = t * (us / vs)^(t - 1)
    q = (1 - t) * (us / vs)^t
    expr = JuMP.@expression(opt.oa_model, p * u + q * v - sign(ws) * w)
    return add_cut(expr, opt)
end
