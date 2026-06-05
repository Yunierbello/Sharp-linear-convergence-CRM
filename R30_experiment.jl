#!/usr/bin/env julia
"""
R^30 experiment of Section 8.4.

Random subspaces `U, V ⊆ R^30` with `dim U = 5`, `dim V = 8`,
`dim(U ∩ V) = 2`. Verifies the rate hierarchy

    ρ_Cheb < ρ_V < c_F² < c_F.
"""

include(joinpath(@__DIR__, "CRMCore.jl"))
using .CRMCore
using LinearAlgebra, Random, Printf


function main()
    rng = MersenneTwister(42)
    P_U, P_V = random_subspace_pair(30, 5, 8, 2; rng=rng)

    angles = principal_angles(P_U, P_V)
    positive = filter(>(1e-6), angles)
    theta_F = positive[1]
    theta_p = min(positive[end], pi / 2)
    a, b = sin(theta_F)^2, sin(theta_p)^2

    rho_V = (b - a) / (a + b)
    c_F = cos(theta_F)
    rho_Cheb = (sqrt(b) - sqrt(a)) / (sqrt(b) + sqrt(a))

    println("R^30 experiment (Section 8.4)")
    println("  dim U = 5, dim V = 8, dim(U ∩ V) = 2")
    println("  All principal angles: ", angles)
    @printf "  theta_F = %.6f\n" theta_F
    @printf "  theta_p = %.6f\n" theta_p
    @printf "  a = sin^2(theta_F) = %.6f\n" a
    @printf "  b = sin^2(theta_p) = %.6f\n" b
    println()
    println("Rate hierarchy:")
    @printf "  rho_Cheb = %.6f\n" rho_Cheb
    @printf "  rho_V    = %.6f\n" rho_V
    @printf "  c_F^2    = %.6f\n" (c_F^2)
    @printf "  c_F      = %.6f\n" c_F
    println("  Hierarchy satisfied: ", rho_Cheb < rho_V < c_F^2 < c_F)
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
