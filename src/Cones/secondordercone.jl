#=
second-order cone
(u, w) : u ≥ ‖w‖
=#

mutable struct SecondOrderConeCache <: ConeCache
    cone::MOI.SecondOrderCone
    oa_model::JuMP.Model
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

function add_init_cuts(cache::SecondOrderConeCache, oa_model::JuMP.Model)
    # cuts are (1, 0) and (1, eᵢ), (1, -eᵢ), ∀i
    u = cache.s_oa[1]
    @views w = cache.s_oa[2:end] # TODO cache?
    d = cache.d
    JuMP.@constraints(oa_model, begin
        u >= 0
        [i in 1:d], u >= w[i]
        [i in 1:d], u >= -w[i]
    end)
    return 1 + 2d
end

function get_subp_cuts(z::Vector{Float64}, cache::SecondOrderConeCache, oa_model::JuMP.Model)
    # strengthened cut is (‖r‖, r)
    # @views r = cache.z[2:end]
    @views r = z[2:end]
    z = vcat(LinearAlgebra.norm(r), r)
    clean_array!(z) && return JuMP.AffExpr[]
    return [dot_expr(z, cache.s_oa, oa_model)]
end

function get_sep_cuts(s::Vector{Float64}, cache::SecondOrderConeCache, oa_model::JuMP.Model)
    # s = cache.s
    us = s[1]
    @views ws = s[2:end]
    ws_norm = LinearAlgebra.norm(ws)
    # check s ∉ K
    if us - ws_norm > -1e-7 # TODO option
        return JuMP.AffExpr[]
    end

    # cut is (1, -ws / ‖ws‖)
    z = vcat(1, -inv(ws_norm) * ws)
    clean_array!(z) && return JuMP.AffExpr[]
    return [dot_expr(z, cache.s_oa, oa_model)]
end
