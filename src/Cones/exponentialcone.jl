#=
exponential cone
(u, v, w) : (u ≤ 0, v = 0, w ≥ 0) or (v > 0, w ≥ v * exp(u / v))
equivalently (v > 0, u ≤ v * log(w / v))
dual cone
(p, q, r) : (p = 0, q ≥ 0, r ≥ 0) or (p < 0, r ≥ -p * exp(q / p - 1))
equivalently (p < 0, q ≥ p * (log(r / -p) + 1))
=#

mutable struct ExponentialConeCache <: ConeCache
    cone::MOI.ExponentialCone
    oa_s::Vector{AE}
    s::Vector{Float64}
    ExponentialConeCache() = new()
end

function create_cache(oa_s::Vector{AE}, cone::MOI.ExponentialCone, ::Bool)
    @assert length(oa_s) == 3
    cache = ExponentialConeCache()
    cache.cone = cone
    cache.oa_s = oa_s
    return cache
end

function add_init_cuts(cache::ExponentialConeCache, oa_model::JuMP.Model)
    (u, v, w) = cache.oa_s
    # add variable bounds and some separation cuts from linearizations at p = -1
    r_lin = [1e-3, 1e0, 1e2]
    JuMP.@constraints(oa_model, begin
        v >= 0
        w >= 0
        [r in r_lin], -u - (log(r) + 1) * v + r * w >= 0
    end)
    return 2 + length(r_lin)
end

function get_subp_cuts(
    z::Vector{Float64},
    cache::ExponentialConeCache,
    oa_model::JuMP.Model,
)
    p = z[1]
    r = z[3]
    if p > 0 || r < 0
        # z ∉ K
        @warn("exponential cone dual vector violates variable bounds")
        return AE[]
    end
    q = z[2]
    (u, v, w) = cache.oa_s

    if p > -1e-12
        return AE[]
    elseif r / -p > 1e-8
        # strengthened cut is (p, p * (log(r / -p) + 1), r)
        q = p * (log(r / -p) + 1)
        cut = JuMP.@expression(oa_model, p * u + q * v + r * w)
    elseif q / p < 30
        # strengthened cut is (p, q, -p * exp(q / p - 1))
        r = -p * exp(q / p - 1)
        cut = JuMP.@expression(oa_model, p * u + q * v + r * w)
    else
        return AE[]
    end
    return [cut]
end

function get_sep_cuts(cache::ExponentialConeCache, oa_model::JuMP.Model)
    (us, vs, ws) = cache.s
    if min(ws, vs) < 0
        error("exponential cone point violates variable lower bounds")
    end
    (u, v, w) = cache.oa_s

    # check s ∉ K and add cut
    if vs <= 1e-7
        # vs near zero, so violation is us
        if us >= 1e-7
            if ws <= 1e-12
                error("cannot add separation cut for exponential cone")
            end

            # cut is (-1, -log(ℯ / 2 * us / ws), us / ws / 2)
            r = us / (2 * ws)
            q = -log(ℯ / 2 * r)
            cut = JuMP.@expression(oa_model, -u + q * v + r * w)
        else
            return AE[]
        end
    elseif ws / vs > 1e-8 && us - vs * log(ws / vs) > 1e-7
        # vs and ws not near zero
        # gradient cut is (-1, log(ws / vs) - 1, vs / ws)
        q = log(ws / vs) - 1
        cut = JuMP.@expression(oa_model, -u + q * v + vs / ws * w)
    elseif vs * exp(us / vs) - ws > 1e-7
        # gradient cut is (-exp(us / vs), (us - vs) / vs * exp(us / vs), 1)
        p = -exp(us / vs)
        q = (us - vs) / vs * p
        cut = JuMP.@expression(oa_model, p * u + q * v + w)
    else
        return AE[]
    end
    return [cut]
end
