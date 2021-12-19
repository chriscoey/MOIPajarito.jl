#=
second-order cone
(u, w) : u ≥ ‖w‖

extended formulation
(u, w) : u ≥ 0, ∃ ϕ ≥ 0, u ≥ 2 Σᵢ ϕᵢ, 2 u ϕᵢ ≥ wᵢ²
linear and 3-dim rotated second order cone constraints
=#

mutable struct SecondOrderConeCache{E <: Extender} <: ConeCache
    cone::MOI.SecondOrderCone
    oa_s::Vector{AE}
    s::Vector{Float64}
    d::Int
    ϕ::Vector{VR}
    SecondOrderConeCache{E}() where {E <: Extender} = new{E}()
end

function create_cache(oa_s::Vector{AE}, cone::MOI.SecondOrderCone, extend::Bool)
    dim = MOI.dimension(cone)
    @assert dim == length(oa_s)
    d = dim - 1
    E = extender(extend, d)
    cache = SecondOrderConeCache{E}()
    cache.cone = cone
    cache.oa_s = oa_s
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
        return AE[]
    end

    # cut is (1, -ws / ‖ws‖)
    r = -inv(ws_norm) * ws
    return _get_cuts(r, cache, oa_model)
end

# unextended formulation

function add_init_cuts(cache::SecondOrderConeCache{Unextended}, oa_model::JuMP.Model)
    u = cache.oa_s[1]
    @views w = cache.oa_s[2:end] # TODO cache?
    d = cache.d
    # u ≥ 0, u ≥ |wᵢ|
    JuMP.@constraints(oa_model, begin
        u >= 0
        [i in 1:d], u >= w[i]
        [i in 1:d], u >= -w[i]
    end)
    return 1 + 2d
end

function _get_cuts(
    r::Vector{Float64},
    cache::SecondOrderConeCache{Unextended},
    oa_model::JuMP.Model,
)
    # strengthened cut is (‖r‖, r)
    clean_array!(r) && return AE[]
    p = LinearAlgebra.norm(r)
    u = cache.oa_s[1]
    @views w = cache.oa_s[2:end]
    cut = JuMP.@expression(oa_model, p * u + JuMP.dot(r, w))
    return [cut]
end

# extended formulation

num_ext_variables(cache::SecondOrderConeCache{Extended}) = cache.d

function extend_start(cache::SecondOrderConeCache{Extended}, s_start::Vector{Float64})
    u_start = s_start[1]
    w_start = s_start[2:end]
    @assert u_start - LinearAlgebra.norm(w_start) >= -1e-7 # TODO
    if u_start < 1e-8
        return zeros(cache.d)
    end
    return [w_i / 2u_start * w_i for w_i in w_start]
end

function setup_auxiliary(cache::SecondOrderConeCache{Extended}, oa_model::JuMP.Model)
    @assert cache.d >= 2
    ϕ = cache.ϕ = JuMP.@variable(oa_model, [1:(cache.d)], lower_bound = 0)
    u = cache.oa_s[1]
    JuMP.@constraint(oa_model, u >= 2 * sum(ϕ))
    return ϕ
end

function add_init_cuts(cache::SecondOrderConeCache{Extended}, oa_model::JuMP.Model)
    u = cache.oa_s[1]
    @views w = cache.oa_s[2:end]
    d = cache.d
    ϕ = cache.ϕ
    # u ≥ 0, u ≥ |wᵢ|
    # disaggregated cut on (u, ϕᵢ, wᵢ) is (1, 2, ±2)
    JuMP.@constraints(oa_model, begin
        u >= 0
        [i in 1:d], u + 2 * ϕ[i] + 2 * w[i] >= 0
        [i in 1:d], u + 2 * ϕ[i] - 2 * w[i] >= 0
    end)
    return 1 + 2d
end

function _get_cuts(
    r::Vector{Float64},
    cache::SecondOrderConeCache{Extended},
    oa_model::JuMP.Model,
)
    clean_array!(r) && return AE[]
    p = LinearAlgebra.norm(r)
    u = cache.oa_s[1]
    @views w = cache.oa_s[2:end]
    ϕ = cache.ϕ
    cuts = AE[]
    for i in 1:(cache.d)
        r_i = r[i]
        iszero(r_i) && continue
        # strengthened disaggregated cut on (u, ϕᵢ, wᵢ) is (rᵢ² / 2‖r‖, ‖r‖, rᵢ)
        cut = JuMP.@expression(oa_model, r_i^2 / 2p * u + p * ϕ[i] + r_i * w[i])
        push!(cuts, cut)
    end
    return cuts
end
