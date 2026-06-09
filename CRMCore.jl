"""
    CRMCore

Core operators for the CRM rate analysis.

References to equations and theorems use the numbering of:
Y. Bello-Cruz. On the sharp linear convergence rate of the
circumcentered--reflection method on subspaces. ArXiv:2606.07888, 2026.
"""
module CRMCore

using LinearAlgebra, Random

export projection_matrix, reflection_matrix, principal_angles,
       circumcenter_three_points, C_T, C_T_via_line_search,
       random_subspace_pair


"""
    projection_matrix(W) -> Matrix

Orthogonal projection onto the column space of `W` (full-rank columns).
"""
function projection_matrix(W::AbstractMatrix)
    F = qr(W)
    k = size(W, 2)
    Q = Matrix(F.Q)[:, 1:k]   # thin Q (n x k)
    return Q * Q'
end


"""
    reflection_matrix(P) -> Matrix

Reflection across the range of the projection `P`: `R = 2P - I`.
"""
function reflection_matrix(P::AbstractMatrix)
    return 2 * P - I
end


"""
    principal_angles(P_U, P_V) -> Vector

Principal angles between `range(P_U)` and `range(P_V)`, sorted ascending.

Returns `p = min(dim U, dim V)` principal angles in `[0, π/2]`.
"""
function principal_angles(P_U::AbstractMatrix, P_V::AbstractMatrix)
    # The principal cosines are the singular values of the operator
    # P_U|_V : V -> U. Equivalently, take orthonormal bases Q_U of U
    # and Q_V of V, then the SVD of Q_U' * Q_V gives the cosines of
    # the principal angles.
    F_U = svd(P_U)
    F_V = svd(P_V)
    dim_U = count(>(0.5), F_U.S)
    dim_V = count(>(0.5), F_V.S)
    Q_U = F_U.U[:, 1:dim_U]
    Q_V = F_V.U[:, 1:dim_V]
    cos_thetas = svdvals(Q_U' * Q_V)
    cos_thetas = clamp.(cos_thetas, 0.0, 1.0)
    return sort(acos.(cos_thetas))
end


"""
    circumcenter_three_points(a, b, c; tol=1e-14) -> Vector

Circumcenter of three points `a, b, c` in `R^n`.

Uses the BBS 2018 closed form (Definition 2.2 in the paper): the circumcenter
lies in `aff{a, b, c}`, equidistant from all three. Reduces to a 2x2
linear system for the affine coefficients.
"""
function circumcenter_three_points(a::AbstractVector, b::AbstractVector,
                                   c::AbstractVector; tol::Real=1e-14)
    s_b = b - a
    s_c = c - a
    G = [dot(s_b, s_b)  dot(s_b, s_c);
         dot(s_c, s_b)  dot(s_c, s_c)]
    rhs = 0.5 .* [dot(s_b, s_b), dot(s_c, s_c)]
    if det(G) < tol
        # Degenerate (collinear or coincident); use midpoint convention.
        other = norm(b - a) >= norm(c - a) ? b : c
        return 0.5 .* (a + other)
    end
    alpha = G \ rhs
    return a + alpha[1] .* s_b + alpha[2] .* s_c
end


"""
    C_T(x, P_U, P_V) -> Vector

Geometric circumcentered-reflection operator `C_T(x) = circ{x, R_U x, R_V R_U x}`.
"""
function C_T(x::AbstractVector, P_U::AbstractMatrix, P_V::AbstractMatrix)
    R_U = reflection_matrix(P_U)
    R_V = reflection_matrix(P_V)
    y = R_U * x
    z = R_V * y
    return circumcenter_three_points(x, y, z)
end


"""
    C_T_via_line_search(v, P_U, P_V) -> Vector

`C_T(v)` for `v ∈ V`, computed via the line-search formula (Theorem 3.4):

    C_T(v) = v + μ_v (T(v) - v),   μ_v = dist²(v, U) / ‖T(v) - v‖².

This is the parameter-free formulation that achieves `ρ_V`.
"""
function C_T_via_line_search(v::AbstractVector, P_U::AbstractMatrix,
                              P_V::AbstractMatrix)
    T_v = P_V * (P_U * v)
    diff = T_v - v
    n2 = dot(diff, diff)
    if n2 < 1e-30
        return copy(v)
    end
    dist2 = norm(v - P_U * v)^2
    mu = dist2 / n2
    return v + mu .* diff
end


"""
    random_subspace_pair(n, dim_U, dim_V, dim_int=0; rng=Random.default_rng())
        -> (P_U, P_V)

Generate subspaces `U, V ⊆ R^n` with prescribed dimensions and
intersection dimension. Returns the projection matrices `(P_U, P_V)`.
"""
function random_subspace_pair(n::Integer, dim_U::Integer, dim_V::Integer,
                              dim_int::Integer=0;
                              rng::AbstractRNG=Random.default_rng())
    # Common subspace
    W = dim_int > 0 ? randn(rng, n, dim_int) : zeros(n, 0)
    # Extra parts
    A = randn(rng, n, dim_U - dim_int)
    B = randn(rng, n, dim_V - dim_int)
    U = hcat(W, A)
    V = hcat(W, B)
    return projection_matrix(U), projection_matrix(V)
end

end  # module CRMCore
