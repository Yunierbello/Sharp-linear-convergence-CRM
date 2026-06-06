#!/usr/bin/env julia
"""
Numerical experiments for Section 8 of:

    Y. Bello-Cruz. On the sharp linear convergence rate of the
    circumcentered--reflection method on subspaces. Submitted 2026.

Julia translation of `numerical_experiments.py`. Section, theorem,
table, and figure numbers below refer to that submitted version.
Reproduces the experiments behind:

  (A) verification of the sharp rate ρ_V             (Section 8.4);
  (B) iteration-count comparison, Table 6            (Section 8.5);
  (C) empirical asymptotic rate of Strategy B        (Section 8.5);
  (D) residual histories for the convergence plot    (Figure 3);
  (E) parameter-free `C_T` on `V` vs parameter-tuned AAMR with the
      optimal `β` of Aragón Artacho and Campoy (2019)
                                                      (Section 8.6,
      Table 8).

Blocks (B), (D), and (E) use the deterministic prescribed-angle pairs
and therefore agree with the Python script exactly. Blocks (A) and (C)
draw random subspaces; with Julia's random number generator the
individual draws differ from the Python run, but the reported
statistics are equivalent.
"""

include(joinpath(@__DIR__, "CRMCore.jl"))
using .CRMCore
using LinearAlgebra, Random, Statistics, Printf


# ----------------------------------------------------------------------
# subspace constructions
# ----------------------------------------------------------------------
"""
    prescribed_pair(thetas) -> (P_U, P_V, n, p)

Subspace pair in `R^{2p}` realising exactly the principal angles `thetas`.
"""
function prescribed_pair(thetas::AbstractVector)
    p = length(thetas)
    n = 2p
    V = zeros(n, p)
    U = zeros(n, p)
    for i in 1:p
        V[i, i] = 1.0
        U[i, i] = cos(thetas[i])
        U[p + i, i] = sin(thetas[i])
    end
    return projection_matrix(U), projection_matrix(V), n, p
end


"""
    random_pair(n, dU, dV, dint, rng) -> (P_U, P_V)

Random subspace pair with prescribed dimensions and intersection.
"""
function random_pair(n::Integer, dU::Integer, dV::Integer,
                     dint::Integer, rng::AbstractRNG)
    W = dint > 0 ? randn(rng, n, dint) : zeros(n, 0)
    A = randn(rng, n, dU - dint)
    B = randn(rng, n, dV - dint)
    return projection_matrix(hcat(W, A)), projection_matrix(hcat(W, B))
end


"""
    exact_intersection(P_U, P_V) -> Matrix

Exact orthogonal projection onto `U ∩ V`. A vector `w` lies in
`U ∩ V` iff `(P_U + P_V) w = 2 w`, so `U ∩ V` is the eigenspace of
the symmetric matrix `P_U + P_V` at eigenvalue 2.
"""
function exact_intersection(P_U::AbstractMatrix, P_V::AbstractMatrix)
    F = eigen(Symmetric(P_U + P_V))
    cols = F.vectors[:, F.values .> 2.0 - 1e-8]
    return size(cols, 2) == 0 ? zero(P_U) : cols * cols'
end


"""
    worst_ray(P_U, P_V) -> (vstar, a, b)

Worst-case ray `v*` and the extreme eigenvalues `(a, b)` of the
operator `M = (I - P_V P_U)|_V`.
"""
function worst_ray(P_U::AbstractMatrix, P_V::AbstractMatrix)
    n = size(P_U, 1)
    F = svd(P_V)
    QV = F.U[:, 1:count(>(0.5), F.S)]
    M = QV' * ((I - P_V * P_U) * QV)
    M = Symmetric(0.5 * (M + M'))
    E = eigen(M)
    w = clamp.(E.values, 0.0, 1.0)
    iF = findfirst(>(1e-9), w)
    a, b = w[iF], w[end]
    vstar = sqrt(b) .* (QV * E.vectors[:, iF]) .+ sqrt(a) .* (QV * E.vectors[:, end])
    return vstar, a, b
end


# ----------------------------------------------------------------------
# (A) verification of the sharp rate -- Section 8.4
# ----------------------------------------------------------------------
function block_A()
    println("="^64)
    println("(A) verification of the sharp rate rho_V   (Section 8.4)")
    rng = MersenneTwister(2024)
    configs = [(20, 8, 6, 2), (30, 5, 8, 2), (40, 12, 10, 3),
               (60, 15, 20, 5), (80, 25, 18, 4)]
    err_star = Float64[]
    excess = Float64[]
    npairs = 0
    for (n, dU, dV, dint) in configs
        for _ in 1:80
            P_U, P_V = random_pair(n, dU, dV, dint, rng)
            vs, a, b = worst_ray(P_U, P_V)
            b - a < 1e-6 && continue
            rv = (b - a) / (a + b)
            P_int = exact_intersection(P_U, P_V)
            vb = P_int * vs
            meas = norm(C_T(vs, P_U, P_V) - vb) / norm(vs - vb)
            push!(err_star, abs(rv - meas))
            for _ in 1:200
                v = P_V * randn(rng, n)
                vb2 = P_int * v
                if norm(v - vb2) > 1e-9
                    r = norm(C_T(v, P_U, P_V) - vb2) / norm(v - vb2)
                    push!(excess, r - rv)
                end
            end
            npairs += 1
        end
    end
    @printf "  pairs tested           = %d\n" npairs
    @printf "  random rays sampled    = %d\n" length(excess)
    @printf "  max |rho_V - rate(v*)| = %.3e\n" maximum(err_star)
    @printf "  max (rate(v) - rho_V)  = %.3e  (<= 0: bound holds)\n" maximum(excess)
end


# ----------------------------------------------------------------------
# iteration drivers
# ----------------------------------------------------------------------
"""
    iters_to_tol(residuals, tol) -> Int

Number of iterations after which the relative residual first drops
to `tol` or below (`residuals[1]` is the initial residual).
"""
function iters_to_tol(residuals::AbstractVector, tol::Real)
    for i in 1:length(residuals)
        residuals[i] <= tol && return i - 1
    end
    return length(residuals) - 1
end


function run_CRM_on_V(v0, P_U, P_V, xbar, K)
    v = copy(v0)
    d0 = norm(v0 - xbar)
    res = [1.0]
    for _ in 1:K
        v = C_T_via_line_search(v - xbar, P_U, P_V) + xbar
        push!(res, norm(v - xbar) / d0)
    end
    return res
end


function run_MAP(v0, P_U, P_V, xbar, K)
    T = P_V * P_U
    v = copy(v0)
    d0 = norm(v0 - xbar)
    res = [1.0]
    for _ in 1:K
        v = T * v
        push!(res, norm(v - xbar) / d0)
    end
    return res
end


function run_Smu(v0, P_U, P_V, xbar, mu, K)
    T = P_V * P_U
    v = copy(v0)
    d0 = norm(v0 - xbar)
    res = [1.0]
    for _ in 1:K
        v = (1.0 - mu) .* v + mu .* (T * v)
        push!(res, norm(v - xbar) / d0)
    end
    return res
end


function run_DRM(x0, P_U, P_V, K)
    T_DR = 0.5 * (Matrix{Float64}(I, size(P_U)...)
                  + reflection_matrix(P_V) * reflection_matrix(P_U))
    z = copy(x0)
    for _ in 1:4000
        z = T_DR * z
    end
    zstar = copy(z)
    z = copy(x0)
    d0 = norm(x0 - zstar)
    res = [1.0]
    for _ in 1:K
        z = T_DR * z
        push!(res, d0 > 0 ? norm(z - zstar) / d0 : 0.0)
    end
    return res
end


"""
Chebyshev semi-iteration applied to `M = (I - P_V P_U)|_V`, started in `V`.
"""
function run_Chebyshev(w0, P_U, P_V, a, b, K)
    n = size(P_U, 1)
    d = (a + b) / 2.0
    r = (a + b) / (b - a)
    Mw(w) = (I - P_V * P_U) * w
    d0 = norm(w0)
    res = [1.0]
    w_prev = copy(w0)
    w_curr = w0 - Mw(w0) ./ d
    push!(res, norm(w_curr) / d0)
    omega = 2r^2 / (2r^2 - 1)
    for _ in 2:K
        w_next = omega .* (w_curr - Mw(w_curr) ./ d) + (1 - omega) .* w_prev
        push!(res, norm(w_next) / d0)
        w_prev, w_curr = w_curr, w_next
        omega = 4r^2 / (4r^2 - omega)
    end
    return res
end


"""
    run_AAMR(x0, P_U, P_V, z, alpha, beta, K, P_int) -> Vector{Float64}

Averaged alternating modified reflections (AAMR) for the best-approximation
problem; Aragón Artacho and Campoy, 2019. Iterates

```
    x_{k+1} = (1 - alpha) x_k
              + alpha (2 beta P_{V-z} - I) (2 beta P_{U-z} - I) x_k
```

starting from `x0` in `R^n` with translated subspaces `U - z`, `V - z`. Returns
the relative-residual history of the shadow sequence `P_U(x_k + z)`, which
converges to `P_{U ∩ V}(z) = P_int * z`. For subspaces, `P_{U-z}(y) = P_U(y+z) - z`.
"""
function run_AAMR(x0, P_U, P_V, z, alpha, beta, K, P_int)
    xbar = P_int * z
    s0 = P_U * (x0 + z)
    d0 = norm(s0 - xbar)
    d0 < 1e-30 && return [0.0]
    x = copy(x0)
    res = [1.0]
    for _ in 1:K
        Ru = 2.0 * beta * (P_U * (x + z) - z) - x
        RvRu = 2.0 * beta * (P_V * (Ru + z) - z) - Ru
        x = (1.0 - alpha) .* x + alpha .* RvRu
        s = P_U * (x + z)
        push!(res, norm(s - xbar) / d0)
    end
    return res
end


# ----------------------------------------------------------------------
# (B) iteration-count comparison -- Table 6 (Section 8.5)
# ----------------------------------------------------------------------
function block_B()
    println("="^64)
    println("(B) iteration counts to relative residual 1e-12   (Table 6)")
    tol, K = 1e-12, 3000
    grid = [("pi/12", "pi/6", pi / 12, pi / 6),
            ("pi/12", "pi/3", pi / 12, pi / 3),
            ("pi/6", "pi/3", pi / 6, pi / 3),
            ("pi/6", "5pi/12", pi / 6, 5pi / 12),
            ("pi/4", "5pi/12", pi / 4, 5pi / 12)]
    @printf "%8s %8s | %5s %5s %5s %5s %5s | %9s\n" "thetaF" "thetaP" "DRM" "MAP" "Smu*" "CRM" "Cheb" "rho_V"
    for (lF, lP, tF, tp) in grid
        P_U, P_V, n, _ = prescribed_pair([tF, tp])
        a, b = sin(tF)^2, sin(tp)^2
        rv = (b - a) / (a + b)
        mu = 2.0 / (a + b)
        xbar = zeros(n)
        vstar, _, _ = worst_ray(P_U, P_V)
        n_crm = iters_to_tol(run_CRM_on_V(vstar, P_U, P_V, xbar, K), tol)
        n_map = iters_to_tol(run_MAP(vstar, P_U, P_V, xbar, K), tol)
        n_smu = iters_to_tol(run_Smu(vstar, P_U, P_V, xbar, mu, K), tol)
        n_drm = iters_to_tol(run_DRM(vstar, P_U, P_V, K), tol)
        n_che = iters_to_tol(run_Chebyshev(vstar, P_U, P_V, a, b, K), tol)
        @printf "%8s %8s | %5d %5d %5d %5d %5d | %9.6f\n" lF lP n_drm n_map n_smu n_crm n_che rv
    end
end


# ----------------------------------------------------------------------
# (C) empirical asymptotic rate of Strategy B -- Section 8.5
# ----------------------------------------------------------------------
function block_C()
    println("="^64)
    println("(C) Strategy B: empirical asymptotic rate of C_T from R^n")
    rng = MersenneTwister(20260519)
    configs = [(20, 8, 6, 2), (30, 5, 8, 2), (40, 12, 10, 3)]
    K, lo, hi = 1500, 1e-11, 1e-2
    rate_cF = Float64[]
    rate_rv = Float64[]
    total = 0
    for (n, dU, dV, dint) in configs
        for _ in 1:20
            P_U, P_V = random_pair(n, dU, dV, dint, rng)
            ang = filter(>(1e-6), principal_angles(P_U, P_V))
            length(ang) < 2 && continue
            tF = ang[1]
            tp = min(ang[end], pi / 2)
            a, b = sin(tF)^2, sin(tp)^2
            cF = cos(tF)
            rv = (b - a) / (a + b)
            P_int = exact_intersection(P_U, P_V)
            for _ in 1:10
                x0 = randn(rng, n)
                xbar = P_int * x0
                x = copy(x0)
                rseq = [norm(x0 - xbar)]
                for _ in 1:K
                    x = C_T(x, P_U, P_V)
                    push!(rseq, norm(x - xbar))
                end
                r0 = rseq[1]
                ratios = Float64[]
                for k in 1:(length(rseq) - 1)
                    if lo * r0 <= rseq[k] <= hi * r0 && rseq[k] > 0
                        push!(ratios, rseq[k + 1] / rseq[k])
                    end
                end
                length(ratios) < 8 && continue
                asym = mean(ratios[max(1, end - 14):end])
                push!(rate_cF, asym / cF)
                push!(rate_rv, asym / rv)
                total += 1
            end
        end
    end
    @printf "  random starts evaluated         = %d\n" total
    @printf "  emp. rate / c_F   : min %.4f  mean %.4f  max %.4f\n" minimum(rate_cF) mean(rate_cF) maximum(rate_cF)
    @printf "  emp. rate / rho_V : min %.4f  mean %.4f  max %.4f\n" minimum(rate_rv) mean(rate_rv) maximum(rate_rv)
    @printf "  fraction within 5%% of rho_V     = %.4f\n" mean(rate_rv .<= 1.05)
end


# ----------------------------------------------------------------------
# (D) residual histories for the convergence plot -- Figure 3
# ----------------------------------------------------------------------
function block_D()
    println("="^64)
    println("(D) residual histories  (theta_F, theta_p) = (pi/6, 5pi/12)")
    tF, tp = pi / 6, 5pi / 12
    P_U, P_V, n, _ = prescribed_pair([tF, tp])
    a, b = sin(tF)^2, sin(tp)^2
    xbar = zeros(n)
    vstar, _, _ = worst_ray(P_U, P_V)
    K = 36
    hist = [("DRM", run_DRM(vstar, P_U, P_V, K)),
            ("MAP", run_MAP(vstar, P_U, P_V, xbar, K)),
            ("CRM", run_CRM_on_V(vstar, P_U, P_V, xbar, K)),
            ("Cheb", run_Chebyshev(vstar, P_U, P_V, a, b, K))]
    for (name, res) in hist
        head = join((@sprintf("%.3e", r) for r in res[1:6]), ", ")
        @printf "  %4s: %s, ...\n" name head
    end
end


# ----------------------------------------------------------------------
# (E) Parameter-free C_T on V vs parameter-tuned AAMR -- Section 8.6
# ----------------------------------------------------------------------
"""
    block_E()

Iteration-count comparison on a triangular `(theta_F, theta_p)` grid:
parameter-free `C_T` on `V` versus parameter-tuned AAMR at the optimum
of Aragón Artacho and Campoy (2019). Both methods start from the
worst-case ray `v*` of CRM, on which `C_T` contracts at exactly `rho_V`
per step.
"""
function block_E()
    println("="^76)
    println("(E) Parameter-free C_T on V  vs  parameter-tuned AAMR")
    println("    starting from v*; tolerance 1e-10  (Section 8.6, Table 8)")
    tol, K = 1e-10, 4000
    n_F = 11
    grid = Tuple{Float64,Float64}[]
    for iF in 1:n_F
        tF = iF * pi / (2 * (n_F + 1))
        for ip in iF:n_F
            tp = ip * pi / (2 * (n_F + 1))
            push!(grid, (tF, tp))
        end
    end

    @printf "%10s %10s | %9s %9s | %9s %10s | %10s\n" "theta_F/pi" "theta_p/pi" "rho_V" "rho_AAMR" "iter CRM" "iter AAMR" "CRM/AAMR"
    crm_iters = Int[]
    aamr_iters = Int[]
    for (tF, tp) in grid
        P_U, P_V, n, _ = prescribed_pair([tF, tp])
        P_int = exact_intersection(P_U, P_V)
        a, b = sin(tF)^2, sin(tp)^2
        rho_V = (b - a) / (a + b)
        rho_AAMR = (1.0 - sin(tF)) / (1.0 + sin(tF))
        vstar, _, _ = worst_ray(P_U, P_V)
        xbar = P_int * vstar
        n_crm = iters_to_tol(run_CRM_on_V(vstar, P_U, P_V, xbar, K), tol)
        beta_star = 1.0 / (1.0 + sin(tF))
        n_aamr = iters_to_tol(run_AAMR(zeros(n), P_U, P_V, vstar,
                                        1.0, beta_star, K, P_int), tol)
        push!(crm_iters, n_crm)
        push!(aamr_iters, n_aamr)
        ratio = n_crm / max(n_aamr, 1)
        @printf "%10.4f %10.4f | %9.6f %9.6f | %9d %10d | %10.3f\n" (tF / pi) (tp / pi) rho_V rho_AAMR n_crm n_aamr ratio
    end
    println()
    n_pairs = length(grid)
    crm_wins = sum(crm_iters .< aamr_iters)
    aamr_wins = sum(aamr_iters .< crm_iters)
    n_tie = sum(crm_iters .== aamr_iters)
    @printf "  grid size                = %d (theta_F, theta_p) pairs\n" n_pairs
    @printf "  CRM faster (fewer iters) = %d of %d (%.1f%%)\n" crm_wins n_pairs (100.0 * crm_wins / n_pairs)
    @printf "  AAMR faster              = %d of %d (%.1f%%)\n" aamr_wins n_pairs (100.0 * aamr_wins / n_pairs)
    @printf "  tie                      = %d of %d\n" n_tie n_pairs
end


function main()
    block_A()
    block_B()
    block_C()
    block_D()
    block_E()
    println("="^64)
    println("done")
end


# Run when executed as a script.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
