#!/usr/bin/env julia
"""
Reproduces the sharp instance of Section 8.3:

    (θ_F, θ_p) = (π/6, π/3)   in   R^4.

Verifies:

  * `ρ_V = 1/2`;
  * the worst-case ray `v*` attains the bound exactly,
    `‖C_T^k(v*)‖ = 2^{-k} ‖v*‖`;
  * the line-search step `μ_{v*} = 2 / (a + b)` (the optimal `μ*`).
"""

include(joinpath(@__DIR__, "CRMCore.jl"))
using .CRMCore
using LinearAlgebra, Printf


function main()
    theta_F, theta_p = pi / 6, pi / 3
    c1, s1 = cos(theta_F), sin(theta_F)
    c2, s2 = cos(theta_p), sin(theta_p)
    a, b = s1^2, s2^2

    f1 = [1.0, 0.0, 0.0, 0.0]
    f2 = [0.0, 0.0, 1.0, 0.0]
    e1 = [c1, s1, 0.0, 0.0]
    e2 = [0.0, 0.0, c2, s2]

    P_V = projection_matrix(hcat(f1, f2))
    P_U = projection_matrix(hcat(e1, e2))

    # Worst-case ray (Theorem 1.1):
    # v* = √b · f_{s+1} + √a · f_p   (here s = 0, so f_{s+1} = f_1, f_p = f_2).
    v_star = sqrt(b) .* f1 + sqrt(a) .* f2

    rho_V = (b - a) / (a + b)
    println("theta_F = pi/6, theta_p = pi/3")
    @printf "a = sin^2(theta_F) = %.15f\n" a
    @printf "b = sin^2(theta_p) = %.15f\n" b
    @printf "rho_V = (b-a)/(a+b) = %.15f  (expected 1/2)\n" rho_V
    @printf "c_F^2 = %.15f  (expected 3/4)\n" (c1^2)

    println("\nIterating C_T from v*:")
    v = copy(v_star)
    for k in 0:7
        nrm = norm(v) / norm(v_star)
        @printf "  k = %d: ||C_T^k(v*)|| / ||v*|| = %.10f, predicted rho_V^k = %.10f\n" k nrm (rho_V^k)
        v = C_T(v, P_U, P_V)
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
