#!/usr/bin/env julia
"""
Chebyshev semi-iteration comparison (Section 8.4).

Compares the empirical convergence of three iterations on a random
subspace pair `(U, V)`, all started from the same `v_0 ∈ V`:

  * CRM (the parameter-free circumcentered-reflection method,
    rate `ρ_V = (b - a) / (a + b)`);
  * the relaxed-AP family `S_{μ*}` at the optimal constant step
    `μ* = 2 / (a + b)`, with the same rate `ρ_V`
    (Bauschke–Bello-Cruz–Nghia–Phan–Wang, Numer. Algorithms 73, 2016,
    Theorem 3.6);
  * Chebyshev semi-iteration applied to `M = (I - P_V P_U)|_V`,
    with rate `ρ_Cheb = (√b - √a) / (√b + √a)`.

The expected ordering `ρ_Cheb < ρ_V` matches Theorem 8.9 of the paper.

Reference: Hageman & Young, *Applied Iterative Methods*, Ch. 5.
"""

include(joinpath(@__DIR__, "CRMCore.jl"))
using .CRMCore
using LinearAlgebra, Random, Printf


"""
    chebyshev_iterate(P_U, P_V, w0, n_iter, a, b) -> Vector{Float64}

Chebyshev semi-iteration applied to `M = (I - P_V P_U)|_V`.

Three-term recursion (Proposition 8.8):

    w_1 = w_0 - d⁻¹ M w_0
    w_{k+1} = ω_{k+1} (w_k - d⁻¹ M w_k) + (1 - ω_{k+1}) w_{k-1}

with `d = (a + b) / 2`, `r = (a + b) / (b - a)`,
`ω_2 = 2r² / (2r² - 1)`, and `ω_{k+1} = 4r² / (4r² - ω_k)` for `k ≥ 2`.

Returns the sequence of norms `‖w_k‖` for `k = 0, 1, …, n_iter`.
"""
function chebyshev_iterate(P_U::AbstractMatrix, P_V::AbstractMatrix,
                            w0::AbstractVector, n_iter::Integer,
                            a::Real, b::Real)
    d = (a + b) / 2
    r = (a + b) / (b - a)

    n = size(P_U, 1)
    Id_n = Matrix{Float64}(I, n, n)
    Mw(w) = (Id_n - P_V * P_U) * w

    history = Float64[norm(w0)]
    w_prev = copy(w0)
    w_curr = w0 - Mw(w0) ./ d
    push!(history, norm(w_curr))

    omega = 2 * r^2 / (2 * r^2 - 1)
    for _ in 2:n_iter
        w_next = omega .* (w_curr - Mw(w_curr) ./ d) + (1 - omega) .* w_prev
        push!(history, norm(w_next))
        w_prev = w_curr
        w_curr = w_next
        omega = 4 * r^2 / (4 * r^2 - omega)
    end
    return history
end


"""
    S_mu_star_iterate(P_U, P_V, v0, n_iter, mu_star) -> Vector{Float64}

Relaxed-MAP family `S_{μ*}(v) = (1 - μ*) v + μ* P_V P_U v` at the
optimal constant step `μ* = 2 / (a + b)`.

Returns the sequence `‖v_k‖` for `k = 0, 1, …, n_iter`.
"""
function S_mu_star_iterate(P_U::AbstractMatrix, P_V::AbstractMatrix,
                            v0::AbstractVector, n_iter::Integer,
                            mu_star::Real)
    history = Float64[norm(v0)]
    v = copy(v0)
    for _ in 1:n_iter
        v = (1.0 - mu_star) .* v + mu_star .* (P_V * (P_U * v))
        push!(history, norm(v))
    end
    return history
end


function main()
    rng = MersenneTwister(123)
    P_U, P_V = random_subspace_pair(20, 6, 5, 0; rng=rng)
    angles = principal_angles(P_U, P_V)
    positive = filter(>(1e-6), angles)
    theta_F = positive[1]
    theta_p = min(positive[end], pi / 2)
    a, b = sin(theta_F)^2, sin(theta_p)^2

    rho_V = (b - a) / (a + b)
    rho_Cheb = (sqrt(b) - sqrt(a)) / (sqrt(b) + sqrt(a))
    mu_star = 2.0 / (a + b)

    println("CRM vs S_{mu_star} vs Chebyshev comparison")
    @printf "  theta_F  = %.4f, theta_p = %.4f\n" theta_F theta_p
    @printf "  mu_star  = %.4f\n" mu_star
    @printf "  rho_V    = %.4f   (rate of CRM and S_{mu_star})\n" rho_V
    @printf "  rho_Cheb = %.4f   (rate of Chebyshev)\n" rho_Cheb
    @printf "  Speedup ratio rho_V / rho_Cheb = %.4f\n\n" (rho_V / rho_Cheb)

    # Common starting point in V (with P_{U ∩ V}(w_0) = 0 since dim_int = 0).
    n = size(P_V, 1)
    w0 = P_V * randn(rng, n)
    n_iter = 30

    # CRM
    v_crm = copy(w0)
    crm_history = Float64[norm(v_crm)]
    for _ in 1:n_iter
        v_crm = C_T_via_line_search(v_crm, P_U, P_V)
        push!(crm_history, norm(v_crm))
    end

    smu_history = S_mu_star_iterate(P_U, P_V, w0, n_iter, mu_star)
    cheb_history = chebyshev_iterate(P_U, P_V, w0, n_iter, a, b)

    @printf "%3s  %15s  %15s  %15s\n" "k" "||v_k|| (CRM)" "||v_k|| (S_mu*)" "||w_k|| (Cheb)"
    for k in 0:2:n_iter
        @printf "%3d  %15.4e  %15.4e  %15.4e\n" k crm_history[k + 1] smu_history[k + 1] cheb_history[k + 1]
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
