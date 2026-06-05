#!/usr/bin/env julia
"""
Floor analysis of Section 8.2.

Computes the optimal rate `Γ*` of the linear two-parameter family
`C_{T, γ, β}` over a grid of `(θ_F, θ_p)` at fixed `θ_p = π/2`,
illustrating the floor `Γ* ≥ 1/3` for `θ_F ≥ ~0.78`.

Generates the comparison plot `ρ_V` vs `Γ*` vs `ρ_Cheb` vs `c_F²`.

Requires the `Plots` and `LaTeXStrings` packages (declared in the
companion `Project.toml`). Activate the project before running:

    julia --project=. floor_analysis.jl
"""

using LinearAlgebra, Printf, Plots, LaTeXStrings


"""
    Gamma_star_grid(theta_F_grid, theta_p) -> Vector

For each `θ_F` in `theta_F_grid`, compute the optimal `Γ*` over
`(γ, β) ∈ (0, 1)²` with `γ + β < 1`.
"""
function Gamma_star_grid(theta_F_grid::AbstractVector, theta_p::Real)
    a = sin.(theta_F_grid) .^ 2
    b = sin(theta_p)^2
    Gamma_star = similar(theta_F_grid, Float64)
    for i in eachindex(theta_F_grid)
        # Brute-force grid search over (γ, β); for production code,
        # replace with a proper minimiser (e.g. Optim.jl).
        ai, bi = a[i], b
        best = 1.0
        for g in range(0.01, 0.99; length=80)
            for be in range(0.01, 0.99 - g; length=80)
                if g + be >= 1
                    continue
                end
                # Spectrum of C_{T, γ, β} on the 2D blocks at angle θ_k:
                # eigenvalues f_0, f_1 = 1 - γ - 2 β sin²(θ) ± √D,
                # where D = γ² - β² sin²(2θ);
                # plus parasitic 1 - 2γ, 1 - 2β, 1 - 2γ - 2β.
                rates = Float64[]
                for theta in (theta_F_grid[i], theta_p)
                    s2 = sin(theta)^2
                    s2_double = sin(2 * theta)^2
                    D = g^2 - be^2 * s2_double
                    if D >= 0
                        f0 = 1 - g - 2 * be * s2 + sqrt(D)
                        f1 = 1 - g - 2 * be * s2 - sqrt(D)
                        push!(rates, abs(f0))
                        push!(rates, abs(f1))
                    else
                        mag = sqrt(1 - 2 * g + 4 * be * s2 * (be + g - 1))
                        push!(rates, mag)
                    end
                end
                push!(rates, abs(1 - 2 * g))
                push!(rates, abs(1 - 2 * be))
                push!(rates, abs(1 - 2 * g - 2 * be))
                rate = maximum(rates)
                if rate < best
                    best = rate
                end
            end
        end
        Gamma_star[i] = best
    end
    return Gamma_star
end


function main()
    # Coarse grid for reasonable runtime.
    theta_F_grid = collect(range(0.1, 1.5; length=15))
    theta_p = pi / 2

    a = sin.(theta_F_grid) .^ 2
    b = sin(theta_p)^2
    rho_V = (b .- a) ./ (a .+ b)
    c_F_sq = cos.(theta_F_grid) .^ 2
    rho_Cheb = (sqrt(b) .- sqrt.(a)) ./ (sqrt(b) .+ sqrt.(a))

    println("Computing Gamma* over $(length(theta_F_grid)) values of theta_F...")
    println("(this takes ~1-2 minutes; reduce grid size if too slow)")
    Gamma_star = Gamma_star_grid(theta_F_grid, theta_p)

    println()
    @printf "%8s  %10s  %10s  %10s  %10s\n" "theta_F" "rho_Cheb" "Gamma*" "rho_V" "c_F^2"
    for i in eachindex(theta_F_grid)
        @printf "%8.4f  %10.4f  %10.4f  %10.4f  %10.4f\n" theta_F_grid[i] rho_Cheb[i] Gamma_star[i] rho_V[i] c_F_sq[i]
    end
    @printf "\nFloor verification: min Gamma* = %.4f, expected ~1/3 = %.4f\n" minimum(Gamma_star) (1 / 3)

    # Plot: ρ_Cheb, Γ*, ρ_V, c_F² vs θ_F at θ_p = π/2.
    # Reproduces the floor-1/3 figure of Section 8.2.
    # Colors are the Okabe-Ito palette (colour-blind friendly).
    p = plot(theta_F_grid, c_F_sq;
             label = L"c_F^2",
             color = "#0072B2",
             linewidth = 1.6,
             size = (700, 450),
             xlabel = L"\theta_F\;\;(\mathrm{radians})",
             ylabel = "rate",
             title = L"\textrm{Floor-}1/3\textrm{ analysis at }\theta_p = \pi/2",
             xlims = (0.0, pi / 2),
             ylims = (0.0, 1.0),
             grid = true,
             legend = :topright)
    plot!(p, theta_F_grid, rho_V;
          label = L"\rho_V",
          color = "#D55E00",
          linewidth = 1.6)
    plot!(p, theta_F_grid, Gamma_star;
          label = L"\Gamma^{\star}",
          color = "#009E73",
          linewidth = 1.6,
          marker = :circle,
          markersize = 4)
    plot!(p, theta_F_grid, rho_Cheb;
          label = L"\rho_{\mathrm{Cheb}}",
          color = "#CC79A7",
          linewidth = 1.6)
    hline!(p, [1 / 3];
           color = :black,
           linestyle = :dash,
           linewidth = 0.9,
           label = L"1/3\;\textrm{floor}")
    mkpath(joinpath(@__DIR__, "figures"))
    out_path = joinpath(@__DIR__, "figures", "floor_one_third.pdf")
    savefig(p, out_path)
    println("\nSaved plot to $(out_path)")
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
