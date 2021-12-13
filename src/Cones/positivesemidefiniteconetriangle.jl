#=
positive semidefinite cone triangle (unscaled)
w ⪰ 0
=#

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
    return mat
end

function add_init_cuts(
    opt::Optimizer,
    s_vars::Vector{VR},
    cone::MOI.PositiveSemidefiniteConeTriangle,
)
    # cuts enforce W_ii ≥ 0, ∀i
    # cuts on (W_ii, W_jj, W_ij) are (1, 1, ±2), ∀i ≂̸ j (a linearization of W_ii * W_jj ≥ W_ij^2)
    # initial OA polyhedron is the dual cone of diagonally dominant matrices
    d = cone.side_dimension
    W = vec_to_symm(d, s_vars)
    JuMP.@constraints(
        opt.oa_model,
        begin
            [i in 1:d], W[i, i] >= 0
            [j in 1:d, i in 1:(j - 1)], W[i, i] + W[j, j] + 2 * W[i, j] >= 0
            [j in 1:d, i in 1:(j - 1)], W[i, i] + W[j, j] - 2 * W[i, j] >= 0
        end
    )
    return d^2
end

function add_subp_cuts(
    opt::Optimizer,
    z::Vector{Float64},
    s_vars::Vector{VR},
    cone::MOI.PositiveSemidefiniteConeTriangle,
)
    # strengthened cuts from eigendecomposition are λᵢ * rᵢ * rᵢ'
    d = cone.side_dimension
    R = vec_to_symm(d, z)
    F = LinearAlgebra.eigen!(LinearAlgebra.Symmetric(R, :U), 1e-10, Inf) # TODO tune
    if isempty(F.values)
        return 0
    end

    R_eig = F.vectors * LinearAlgebra.Diagonal(sqrt.(F.values))
    W = vec_to_symm(d, s_vars)
    num_cuts = 0
    for i in eachindex(F.values)
        @views r_i = R_eig[:, i]
        R_i = r_i * r_i'
        clean_array!(R_i) && continue
        expr = dot_expr(R_i, W, opt)
        num_cuts += add_cut(expr, opt)
    end
    return num_cuts
end

function add_sep_cuts(
    opt::Optimizer,
    s::Vector{Float64},
    s_vars::Vector{VR},
    cone::MOI.PositiveSemidefiniteConeTriangle,
)
    # check s ∉ K
    d = cone.side_dimension
    sW = vec_to_symm(d, s)
    F = LinearAlgebra.eigen!(LinearAlgebra.Symmetric(sW, :U), -Inf, -opt.tol_feas)
    if isempty(F.values)
        return 0
    end

    # separation cuts from eigendecomposition are wᵢ * wᵢ'
    W = vec_to_symm(d, s_vars)
    num_cuts = 0
    for i in eachindex(F.values)
        @views w_i = F.vectors[:, i]
        W_i = w_i * w_i'
        clean_array!(W_i) && continue
        expr = dot_expr(W_i, W, opt)
        num_cuts += add_cut(expr, opt)
    end
    return num_cuts
end
