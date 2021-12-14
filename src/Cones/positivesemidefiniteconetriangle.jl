#=
positive semidefinite cone triangle (unscaled)
w ⪰ 0
=#

mutable struct PosSemiDefConeCache <: ConeCache
    cone::MOI.PositiveSemidefiniteConeTriangle
    s_oa::Vector{VR}
    s::Vector{Float64}
    d::Int
    W::Matrix{VR}
    PosSemiDefConeCache() = new()
end

function create_cache(s_oa::Vector{VR}, cone::MOI.PositiveSemidefiniteConeTriangle, ::Bool)
    cache = PosSemiDefConeCache()
    cache.cone = cone
    cache.s_oa = s_oa
    cache.d = cone.side_dimension
    cache.W = vec_to_symm(cache.d, s_oa)
    return cache
end

function add_init_cuts(cache::PosSemiDefConeCache, oa_model::JuMP.Model)
    # cuts enforce W_ii ≥ 0, ∀i
    # cuts on (W_ii, W_jj, W_ij) are (1, 1, ±2), ∀i ≂̸ j (a linearization of W_ii * W_jj ≥ W_ij^2)
    # initial OA polyhedron is the dual cone of diagonally dominant matrices
    d = cache.d
    W = cache.W
    for i in 1:d
        JuMP.set_lower_bound(W[i, i], 0)
    end
    JuMP.@constraints(
        oa_model,
        begin
            [j in 1:d, i in 1:(j - 1)], W[i, i] + W[j, j] + 2 * W[i, j] >= 0
            [j in 1:d, i in 1:(j - 1)], W[i, i] + W[j, j] - 2 * W[i, j] >= 0
        end
    )
    return d^2
end

function get_subp_cuts(z::Vector{Float64}, cache::PosSemiDefConeCache, oa_model::JuMP.Model)
    # strengthened cuts from eigendecomposition are λᵢ * rᵢ * rᵢ'
    R = vec_to_symm(cache.d, z)
    F = LinearAlgebra.eigen!(LinearAlgebra.Symmetric(R, :U), 1e-10, Inf) # TODO tune
    isempty(F.values) && return JuMP.AffExpr[]
    R_eig = F.vectors * LinearAlgebra.Diagonal(sqrt.(F.values))
    return _get_cuts(R_eig, cache, oa_model)
end

function get_sep_cuts(cache::PosSemiDefConeCache, oa_model::JuMP.Model)
    # check s ∉ K
    sW = vec_to_symm(cache.d, cache.s)
    F = LinearAlgebra.eigen!(LinearAlgebra.Symmetric(sW, :U), -Inf, -1e-7)
    isempty(F.values) && return JuMP.AffExpr[]
    return _get_cuts(F.vectors, cache, oa_model)
end

function _get_cuts(R_eig::Matrix{Float64}, cache::PosSemiDefConeCache, oa_model::JuMP.Model)
    # cuts from eigendecomposition are rᵢ * rᵢ'
    W = cache.W
    cuts = JuMP.AffExpr[]
    for i in size(R_eig, 2)
        @views r_i = R_eig[:, i]
        R_i = r_i * r_i'
        clean_array!(R_i) && continue
        cut = dot_expr(R_i, W, oa_model)
        push!(cuts, cut)
    end
    return cuts
end

function vec_to_symm(d::Int, vec::Vector{T}) where {T}
    mat = Matrix{T}(undef, d, d)
    k = 1
    for j in 1:d
        for i in 1:(j - 1)
            mat[i, j] = mat[j, i] = vec[k]
            k += 1
        end
        mat[j, j] = vec[k]
        k += 1
    end
    @assert k - 1 == length(vec)
    return mat
end
