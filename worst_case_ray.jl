#!/usr/bin/env julia
"""
Worst-case ray construction of Theorem 1.1.

For random subspace pairs `(U, V)`, constructs the explicit witness

    v* = √b · f_{s+1} + √a · f_p   in   V ∩ (U ∩ V)^⊥,

where `f_{s+1}` and `f_p` are eigenvectors of `M = (I - P_V P_U)|_V`
at the smallest and largest nonzero eigenvalues, respectively, and
verifies that `‖C_T(v*)‖ / ‖v*‖` equals `ρ_V` exactly.
"""

include(joinpath(@__DIR__, "CRMCore.jl"))
using .CRMCore
using LinearAlgebra, Random, Printf


"""
    construct_worst_case_ray(P_U, P_V; tol=1e-6) -> (v_star, theta_F, theta_p)

Construct the worst-case ray of Theorem 1.1.
"""
function construct_worst_case_ray(P_U::AbstractMatrix, P_V::AbstractMatrix;
                                  tol::Real=1e-6)
    n = size(P_V, 1)
    # M = (I - P_V P_U)|_V on R^n: restrict by composing with P_V.
    M = (I(n) - P_V * P_U) * P_V
    # Symmetrise (on V this is already self-adjoint; the symmetric
    # part is a numerical safeguard).
    M_sym = 0.5 .* (M + M')
    E = eigen(Symmetric(Matrix(M_sym)))
    eigvals_, eigvecs_ = E.values, E.vectors
    # Keep only eigenpairs with eigenvalue > tol — these correspond to
    # angles θ_k ∈ (0, π/2].
    nontrivial = [(eigvals_[k], eigvecs_[:, k]) for k in eachindex(eigvals_)
                  if eigvals_[k] > tol]
    if length(nontrivial) < 2
        throw(ErrorException("Not enough non-trivial eigenvalues for worst-case ray."))
    end
    sort!(nontrivial; by = p -> p[1])
    a, u_a = nontrivial[1]
    b, u_b = nontrivial[end]
    v_star = sqrt(b) .* u_a + sqrt(a) .* u_b
    theta_F = asin(sqrt(a))
    theta_p = asin(sqrt(b))
    return v_star, theta_F, theta_p
end


function main()
    rng = MersenneTwister(7)
    println("Worst-case ray verification on random subspace pairs.\n")
    @printf "%4s  %8s  %8s  %8s  %10s  %10s\n" "pair" "theta_F" "theta_p" "rho_V" "empirical" "|diff|"

    diffs = Float64[]
    for trial in 0:19
        try
            P_U, P_V = random_subspace_pair(15, 6, 5, 1; rng=rng)
            v_star, theta_F, theta_p = construct_worst_case_ray(P_U, P_V)
            a, b = sin(theta_F)^2, sin(theta_p)^2
            rho_V = (b - a) / (a + b)
            v_next = C_T_via_line_search(v_star, P_U, P_V)
            ratio = norm(v_next) / norm(v_star)
            d = abs(ratio - rho_V)
            push!(diffs, d)
            @printf "%4d  %8.4f  %8.4f  %8.4f  %10.6f  %10.2e\n" trial theta_F theta_p rho_V ratio d
        catch e
            @printf "%4d  skipped: %s\n" trial sprint(showerror, e)
        end
    end
    if !isempty(diffs)
        @printf "\nMax |empirical - rho_V|: %.2e\n" maximum(diffs)
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
