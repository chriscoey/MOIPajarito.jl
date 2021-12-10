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
    # strengthened cut is (‖r‖, r)
    @views r = z[2:end]
    z = vcat(LinearAlgebra.norm(r), r)
    clean_array!(z) && return 0
    expr = dot_expr(z, s_vars, opt)
    return add_cut(expr, opt)
end

function add_sep_cuts(
    opt::Optimizer,
    s::Vector{Float64},
    s_vars::Vector{VR},
    ::MOI.SecondOrderCone,
)
    us = s[1]
    @views ws = s[2:end]
    ws_norm = LinearAlgebra.norm(ws)
    # check s ∉ K
    if us - ws_norm > -opt.tol_feas
        return 0
    end

    # cut is (1, -ws / ‖ws‖)
    z = vcat(1, -inv(ws_norm) * ws)
    clean_array!(z) && return 0
    expr = dot_expr(z, s_vars, opt)
    return add_cut(expr, opt)
end
