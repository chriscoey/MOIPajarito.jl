#=
positive semidefinite cone triangle (unscaled)
w ⪰ 0
=#

mutable struct PositiveSemidefiniteConeTriangleCache <: ConeCache
    cone::MOI.PositiveSemidefiniteConeTriangle
    oa_s::Vector{AE}
    s::Vector{Float64}
    d::Int
    W::Matrix{<:Union{VR, AE}}
    PositiveSemidefiniteConeTriangleCache() = new()
end

function create_cache(oa_s::Vector{AE}, cone::MOI.PositiveSemidefiniteConeTriangle, ::Bool)
    cache = PositiveSemidefiniteConeTriangleCache()
    cache.cone = cone
    cache.oa_s = oa_s
    cache.d = cone.side_dimension
    cache.W = vec_to_symm(cache.d, oa_s)
    return cache
end

function add_init_cuts(cache::PositiveSemidefiniteConeTriangleCache, oa_model::JuMP.Model)
    # cuts enforce W_ii ≥ 0, ∀i
    # cuts on (W_ii, W_jj, W_ij) are (1, 1, ±2), ∀i != j
    # (a linearization of W_ii * W_jj ≥ W_ij^2)
    # initial OA polyhedron is the dual cone of diagonally dominant matrices
    d = cache.d
    W = cache.W
    JuMP.@constraints(
        oa_model,
        begin
            [i in 1:d], W[i, i] >= 0
            [j in 1:d, i in 1:(j - 1)], W[i, i] + W[j, j] + 2 * W[i, j] >= 0
            [j in 1:d, i in 1:(j - 1)], W[i, i] + W[j, j] - 2 * W[i, j] >= 0
        end
    )
    return d^2
end

function get_subp_cuts(
    z::Vector{Float64},
    cache::PositiveSemidefiniteConeTriangleCache,
    oa_model::JuMP.Model,
)
    # strengthened cuts from eigendecomposition are λᵢ * rᵢ * rᵢ'
    R = vec_to_symm(cache.d, z)
    F = LinearAlgebra.eigen!(LinearAlgebra.Symmetric(R, :U), 1e-10, Inf) # TODO tune
    isempty(F.values) && return AE[]
    R_eig = F.vectors * LinearAlgebra.Diagonal(sqrt.(F.values))
    return _get_cuts(R_eig, cache, oa_model)
end

function get_sep_cuts(cache::PositiveSemidefiniteConeTriangleCache, oa_model::JuMP.Model)
    # check s ∉ K
    sW = vec_to_symm(cache.d, cache.s)
    F = LinearAlgebra.eigen!(LinearAlgebra.Symmetric(sW, :U), -Inf, -1e-7)
    isempty(F.values) && return AE[]
    return _get_cuts(F.vectors, cache, oa_model)
end

function _get_cuts(
    R_eig::Matrix{Float64},
    cache::PositiveSemidefiniteConeTriangleCache,
    oa_model::JuMP.Model,
)
    # cuts from eigendecomposition are rᵢ * rᵢ'
    W = cache.W
    cuts = AE[]
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
