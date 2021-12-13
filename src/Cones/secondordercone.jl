#=
second-order cone
(u, w) : u ≥ ‖w‖
=#

mutable struct SecondOrderConeCache <: ConeCache
    cone::MOI.SecondOrderCone
    s_oa::Vector{VR}
    s::Vector{Float64}
    z::Vector{Float64}
    d::Int
    SecondOrderConeCache() = new()
end

function create_cache(
    s_oa::Vector{VR},
    cone::MOI.SecondOrderCone,
    )
    dim = MOI.dimension(cone)
    @assert dim == length(s_oa)
    cache = SecondOrderConeCache()
    cache.cone = cone
    cache.s_oa = s_oa
    cache.d = dim - 1
    return cache
end

function add_init_cuts(opt::Optimizer, cache::SecondOrderConeCache)
    # cuts are (1, 0) and (1, eᵢ), (1, -eᵢ), ∀i
    u = cache.s_oa[1]
    @views w = s_oa[2:end] # TODO cache?
    d = cache.d
    JuMP.@constraints(opt.oa_model, begin
        u >= 0
        [i in 1:d], u >= w[i]
        [i in 1:d], u >= -w[i]
    end)
    return 1 + 2d
end

function add_subp_cuts(opt::Optimizer, cache::SecondOrderConeCache)
    # strengthened cut is (‖r‖, r)
    @views r = cache.z[2:end]
    z = vcat(LinearAlgebra.norm(r), r)
    clean_array!(z) && return 0
    expr = dot_expr(z, cache.s_oa, opt)
    return add_cut(expr, opt)
end

function add_sep_cuts(opt::Optimizer, cache::SecondOrderConeCache)
    s = cache.s
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
    expr = dot_expr(z, cache.s_oa, opt)
    return add_cut(expr, opt)
end
