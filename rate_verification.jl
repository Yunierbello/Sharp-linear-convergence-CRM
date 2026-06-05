#!/usr/bin/env julia
"""
Statistical verification of the sharp rate ρ_V on random subspace pairs.

Theorem 1.1 states that ρ_V is the SHARP ONE-STEP contraction factor
of C_T on V: for every v ∈ V,

    ‖C_T(v) - v̄‖ ≤ ρ_V · ‖v - v̄‖,        v̄ = P_{U ∩ V}(v),

with equality on an explicit worst-case ray. This script checks both
halves of that statement on random subspace pairs (U, V) in R^n:

  * the bound holds — no sampled ray exceeds ρ_V;
  * the bound is sharp — the largest one-step factor over many
    sampled rays approaches ρ_V from below.

The companion script `worst_case_ray.jl` exhibits the explicit ray on
which the factor equals ρ_V exactly; here the same constant is
confirmed statistically, without constructing that ray, and using the
geometric circumcenter C_T directly.

Note: ρ_V is the one-step contraction factor, not the asymptotic
per-iteration rate. The asymptotic rate of C_T from a generic start in
V is start-dependent and strictly smaller than ρ_V, so it is not the
quantity to compare against the closed form.
"""

include(joinpath(@__DIR__, "CRMCore.jl"))
using .CRMCore
using LinearAlgebra, Random, Statistics, Printf


"""
    project_intersection(P_U, P_V; n_iter=400) -> Matrix

Projection `P_{U ∩ V}` as the limit of `(P_V P_U)^k` (von Neumann).
"""
function project_intersection(P_U::AbstractMatrix, P_V::AbstractMatrix;
                              n_iter::Integer=400)
    n = size(P_U, 1)
    M = Matrix{Float64}(I, n, n)
    for _ in 1:n_iter
        M = P_V * (P_U * M)
    end
    return M
end


"""
    one_step_factors(P_U, P_V, rho_V; n_rays=3000, rng) -> (max_factor, max_excess)

Sample `n_rays` random rays `v ∈ V` and, for each, evaluate the
one-step contraction factor of the geometric circumcenter,

    ‖C_T(v) - v̄‖ / ‖v - v̄‖,    v̄ = P_{U ∩ V}(v).

Returns the largest factor observed and the largest signed gap
`factor - ρ_V`, which should stay ≤ 0 up to roundoff.
"""
function one_step_factors(P_U::AbstractMatrix, P_V::AbstractMatrix,
                          rho_V::Real; n_rays::Integer=3000,
                          rng::AbstractRNG=Random.default_rng())
    n = size(P_V, 1)
    P_int = project_intersection(P_U, P_V)
    max_factor = 0.0
    max_excess = -Inf
    for _ in 1:n_rays
        v = P_V * randn(rng, n)
        v_bar = P_int * v
        d = norm(v - v_bar)
        d < 1e-9 && continue
        factor = norm(C_T(v, P_U, P_V) - v_bar) / d
        max_factor = max(max_factor, factor)
        max_excess = max(max_excess, factor - rho_V)
    end
    return max_factor, max_excess
end


function main()
    rng = MersenneTwister(0)
    n = 20
    dim_U, dim_V, dim_int = 8, 6, 2
    N = 50  # number of random subspace pairs

    println("Random subspace pairs in R^$(n): dim U = $(dim_U), dim V = $(dim_V), dim(U ∩ V) = $(dim_int)")
    println("Sampling 3000 rays per pair; $N pairs.\n")
    @printf "%4s  %8s  %8s  %12s  %12s  %12s\n" "pair" "theta_F" "theta_p" "rho_V" "max factor" "max excess"
    excesses = Float64[]
    gaps = Float64[]
    for trial in 0:(N - 1)
        P_U, P_V = random_subspace_pair(n, dim_U, dim_V, dim_int; rng=rng)
        angles = principal_angles(P_U, P_V)
        # θ_F = smallest strictly positive principal angle
        # θ_p = largest principal angle in [0, π/2]
        positive = filter(>(1e-6), angles)
        length(positive) < 2 && continue
        theta_F = positive[1]
        theta_p = min(positive[end], pi / 2)
        a, b = sin(theta_F)^2, sin(theta_p)^2
        rho_V = (a + b) > 0 ? (b - a) / (a + b) : 0.0
        max_factor, max_excess = one_step_factors(P_U, P_V, rho_V; rng=rng)
        push!(excesses, max_excess)
        push!(gaps, rho_V - max_factor)
        if trial < 10
            @printf "%4d  %8.4f  %8.4f  %12.8f  %12.8f  %12.2e\n" trial theta_F theta_p rho_V max_factor max_excess
        end
    end
    println("...")
    println("  (showing first 10 of $N)\n")
    @printf "Max excess over all pairs (factor - rho_V): %11.2e   (<= 0: the bound holds)\n" maximum(excesses)
    @printf "Max gap rho_V - (sampled max factor):       %11.2e   (small: the bound is tight)\n" maximum(gaps)
end


# Run when executed as a script.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
