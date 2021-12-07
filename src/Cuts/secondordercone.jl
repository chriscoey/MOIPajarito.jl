#=
second-order cone
(u, w) : u ≥ ‖w‖

TODO extended formulation

- use a cache for each cone
- maybe define both the unextended and the extended second order cones,
because the functions will get messy without dispatch
- in the construct models function, can construct the list of cone caches,
and any options to Optimizer that determine eg whether to use the SOC EF
can impact these caches then. because not convenient to have to pass
options directly to caches. need to keep Pajarito simple with minimal
options and minimal interfaces! want to be able to test turning on/off EFs



=#

function add_init_cuts(opt::Optimizer, s_vars::Vector{VR}, ::MOI.SecondOrderCone)
    # cuts are (1, 0) and (1, eᵢ), (1, -eᵢ), ∀i
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
    # extreme cut is (‖r‖, r)
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

    # scaled extreme cut is (1, -r / ‖r‖)
    u = s_vars[1]
    @views w = s_vars[2:end]
    d = length(w)
    expr = JuMP.@expression(opt.oa_model, u - sum(r[i] / r_norm * w[i] for i in 1:d))
    return add_cut(expr, opt)
end
