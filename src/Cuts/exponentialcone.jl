#=
exponential cone
(u, v, w) : (u ≤ 0, v = 0, w ≥ 0) or (v > 0, w ≥ v * exp(u / v))
equivalently (v > 0, u ≤ v * log(w / v))
dual cone
(p, q, r) : (p = 0, q ≥ 0, r ≥ 0) or (p < 0, r ≥ -p * exp(q / p - 1))
equivalently (p < 0, q ≥ p * (log(r / -p) + 1))
=#

function add_init_cuts(opt::Optimizer, s_vars::Vector{VR}, ::MOI.ExponentialCone)
    @assert length(s_vars) == 3
    (u, v, w) = s_vars
    # add variable bounds and some separation cuts from linearizations at p = -1
    r_lin = [1e-3, 1e0, 1e2]
    JuMP.@constraints(opt.oa_model, begin
        v >= 0
        w >= 0
        [r in r_lin], -u - (log(r) + 1) * v + r * w >= 0
    end)
    return 2 + length(r_lin)
end

function add_subp_cuts(
    opt::Optimizer,
    z::Vector{Float64},
    s_vars::Vector{VR},
    ::MOI.ExponentialCone,
)
    p = z[1]
    r = z[3]
    if p >= 0 || r <= 0
        # z ∉ K
        @warn("dual vector is not in the dual cone")
        return 0
    end

    # strengthened cut is (p, p * (log(r / -p) + 1), r)
    q = p * (log(r / -p) + 1)
    if r < 1e-9
        # TODO needed for GLPK
        r = 0.0
    end
    (u, v, w) = s_vars
    expr = JuMP.@expression(opt.oa_model, p * u + q * v + r * w)
    return add_cut(expr, opt)
end

function add_sep_cuts(
    opt::Optimizer,
    s::Vector{Float64},
    s_vars::Vector{VR},
    ::MOI.ExponentialCone,
)
    (us, vs, ws) = s
    if min(ws, vs) <= -opt.tol_feas
        error("exp cone point violates initial cuts")
    end
    if min(ws, vs) <= 0
        # error("TODO add a cut")
        return 0
    end

    # check s ∉ K
    if vs * log(ws / vs) - us > -opt.tol_feas
        return 0
    end

    # gradient cut is (-1, log(ws / vs) - 1, ws / vs)
    (u, v, w) = s_vars
    p = log(ws / vs) - 1
    expr = JuMP.@expression(opt.oa_model, -u + p * v + vs / ws * w)
    return add_cut(expr, opt)
end
