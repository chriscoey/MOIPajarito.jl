#=
second-order cone
(u, w) : u ≥ ‖w‖

TODO extended formulation
=#

function add_init_cuts(opt::Optimizer, s_vars::Vector{VR}, ::MOI.SecondOrderCone)
    # duals are (1, 0) and (1, eᵢ), (1, -eᵢ), ∀i
    u = s_vars[1]
    @views w = s_vars[2:end]
    d = length(w)
    JuMP.@constraints(opt.oa_model, begin
        u >= 0
        [i in 1:d], u >= w[i]
        [i in 1:d], u >= -w[i]
    end)

    return 1 + 2d
end

function add_subp_cuts(
    opt::Optimizer,
    z::Vector{Float64},
    s_vars::Vector{VR},
    ::MOI.SecondOrderCone,
)
    # extreme dual is (‖r‖, r)
    @views r = z[2:end]
    u = s_vars[1]
    @views w = s_vars[2:end]
    expr = JuMP.@expression(opt.oa_model, LinearAlgebra.norm(r) * u + JuMP.dot(r, w))
    return add_cut(expr, opt)
end

function add_sep_cuts(
    opt::Optimizer,
    s::Vector{Float64},
    s_vars::Vector{VR},
    ::MOI.SecondOrderCone,
)
    # check (p, r) ∉ K
    p = s[1]
    @views r = s[2:end]
    r_norm = LinearAlgebra.norm(r)
    if p - r_norm > -opt.tol_feas
        return 0
    end

    # scaled extreme dual is (1, -r / ‖r‖)
    u = s_vars[1]
    @views w = s_vars[2:end]
    d = length(w)
    expr = JuMP.@expression(opt.oa_model, u - sum(r[i] / r_norm * w[i] for i in 1:d))
    return add_cut(expr, opt)
end
