#=
second-order cone
(u, w) : u ≥ ‖w‖

extended formulation
(u, w) : u ≥ 0, ∃ ϕ ≥ 0, u ≥ 2 Σᵢ ϕᵢ, 2 u ϕᵢ ≥ wᵢ²
linear and 3-dim rotated second order cone constraints
=#

mutable struct SecondOrderConeCache{E <: Extender} <: ConeCache
    cone::MOI.SecondOrderCone
    s_oa::Vector{VR}
    s::Vector{Float64}
    d::Int
    ϕ::Vector{VR}
    SecondOrderConeCache{E}() where {E <: Extender} = new{E}()
end

function create_cache(s_oa::Vector{VR}, cone::MOI.SecondOrderCone, extend::Bool)
    dim = MOI.dimension(cone)
    @assert dim == length(s_oa)
    d = dim - 1
    E = extender(extend, d)
    cache = SecondOrderConeCache{E}()
    cache.cone = cone
    cache.s_oa = s_oa
    cache.d = d
    return cache
end

function get_subp_cuts(
    z::Vector{Float64},
    cache::SecondOrderConeCache,
    oa_model::JuMP.Model,
)
    return _get_cuts(z[2:end], cache, oa_model)
end

function get_sep_cuts(cache::SecondOrderConeCache, oa_model::JuMP.Model)
    s = cache.s
    us = s[1]
    @views ws = s[2:end]
    ws_norm = LinearAlgebra.norm(ws)
    # check s ∉ K
    if us - ws_norm > -1e-7 # TODO option
        return JuMP.AffExpr[]
    end

    # cut is (1, -ws / ‖ws‖)
    r = -inv(ws_norm) * ws
    return _get_cuts(r, cache, oa_model)
end

# unextended formulation

function add_init_cuts(cache::SecondOrderConeCache{Unextended}, oa_model::JuMP.Model)
    u = cache.s_oa[1]
    @views w = cache.s_oa[2:end] # TODO cache?
    d = cache.d
    # u ≥ 0, u ≥ |wᵢ|
    JuMP.set_lower_bound(u, 0)
    JuMP.@constraints(oa_model, begin
        [i in 1:d], u ≥ w[i]
        [i in 1:d], u ≥ -w[i]
    end)
    return 1 + 2d
end

function _get_cuts(
    r::Vector{Float64},
    cache::SecondOrderConeCache{Unextended},
    oa_model::JuMP.Model,
)
    # strengthened cut is (‖r‖, r)
    clean_array!(r) && return JuMP.AffExpr[]
    p = LinearAlgebra.norm(r)
    u = cache.s_oa[1]
    @views w = cache.s_oa[2:end]
    cut = JuMP.@expression(oa_model, p * u + JuMP.dot(r, w))
    return [cut]
end

# extended formulation

function extend_warm_start(cache::SecondOrderConeCache{Extended})
    s_start = JuMP.start_value.(cache.s_oa)
    u_start = s_start[1]
    ϕ = cache.ϕ
    if u_start < 1e-8
        JuMP.set_start_value.(ϕ, 0)
        return
    end
    for i in 1:(cache.d)
        w_i = s_start[1 + i]
        JuMP.set_start_value(ϕ[i], w_i^2 / 2u_start)
    end
    return
end

function setup_auxiliary(cache::SecondOrderConeCache{Extended}, oa_model::JuMP.Model)
    cache.ϕ = JuMP.@variable(oa_model, [1:(cache.d)], lower_bound = 0)
    u = cache.s_oa[1]
    JuMP.@constraint(oa_model, u ≥ 2 * sum(cache.ϕ))
    return
end

function add_init_cuts(cache::SecondOrderConeCache{Extended}, oa_model::JuMP.Model)
    u = cache.s_oa[1]
    @views w = cache.s_oa[2:end]
    d = cache.d
    ϕ = cache.ϕ
    # u ≥ 0, u ≥ |wᵢ|
    # disaggregated cut on (u, ϕᵢ, wᵢ) is (1, 2, ±2)
    JuMP.set_lower_bound(u, 0)
    JuMP.@constraints(oa_model, begin
        [i in 1:d], u + 2 * ϕ[i] + 2 * w[i] ≥ 0
        [i in 1:d], u + 2 * ϕ[i] - 2 * w[i] ≥ 0
    end)
    return 1 + 2d
end

function _get_cuts(
    r::Vector{Float64},
    cache::SecondOrderConeCache{Extended},
    oa_model::JuMP.Model,
)
    clean_array!(r) && return JuMP.AffExpr[]
    p = LinearAlgebra.norm(r)
    u = cache.s_oa[1]
    @views w = cache.s_oa[2:end]
    ϕ = cache.ϕ
    cuts = JuMP.AffExpr[]
    for i in 1:(cache.d)
        r_i = r[i]
        iszero(r_i) && continue
        # strengthened disaggregated cut on (u, ϕᵢ, wᵢ) is (rᵢ² / 2‖r‖, ‖r‖, rᵢ)
        cut = JuMP.@expression(oa_model, r_i^2 / 2p * u + p * ϕ[i] + r_i * w[i])
        push!(cuts, cut)
    end
    return cuts
end
