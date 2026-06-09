[README.md](https://github.com/user-attachments/files/28619844/README.md)
# Sharp linear convergence rate of CRM on subspaces — Julia code

Julia companion code for

> Y. Bello-Cruz. *On the sharp linear convergence rate of the
> circumcentered-reflection method on subspaces.* 2026.
> [arXiv:XXXX.XXXXX](https://arxiv.org/abs/2606.07888)

For two subspaces *U, V* ⊆ ℝⁿ, the circumcentered-reflection method (CRM) of
Behling, Bello-Cruz, and Santos computes the projection onto *U ∩ V* from the
reflections across *U* and *V*. Initialized in *V*, CRM contracts at the sharp
rate

    ρ_V = (sin²θ_p − sin²θ_F) / (sin²θ_p + sin²θ_F),

with θ_F the Friedrichs angle and θ_p the largest principal angle between *U*
and *V*. The bound is sharp (attained on an explicit ray in *V*) and optimal
among parameter-free single-step iterations. This code verifies ρ_V to machine
precision and reproduces every number and figure in the paper.

## Requirements

Julia ≥ 1.9. All packages are declared in `Project.toml`.

## Setup

From this directory:

```julia
julia> using Pkg; Pkg.activate("."); Pkg.instantiate()
```

or from the shell:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Files

| File | Role |
|---|---|
| `CRMCore.jl` | Core module: projection / reflection matrices, principal angles, circumcenter, `C_T`, line-search. |
| `rate_verification.jl` | Statistical check that `ρ_V` is the sharp one-step contraction factor of `C_T` on random subspace pairs: no sampled ray exceeds `ρ_V`, and the largest sampled factor approaches it. |
| `worst_case_ray.jl` | Construction of the explicit witness ray `v* = √b · f_{s+1} + √a · f_p` and verification that `‖C_T(v*)‖ / ‖v*‖ = ρ_V`. |
| `floor_analysis.jl` | Optimal rate `Γ*` of the linear two-parameter family `C_{T,γ,β}` on a grid `(θ_F, θ_p)`; produces `figures/floor_one_third.pdf` of Section 8.2 (rate comparisons). |
| `sharp_instance.jl` | Explicit `(θ_F, θ_p) = (π/6, π/3)` sharp instance of Section 8.3 (a concrete sharp instance). |
| `R30_experiment.jl` | Concrete `R^30` experiment of Section 8.4 (numerical verification) with `dim U = 5`, `dim V = 8`, `dim(U ∩ V) = 2`. |
| `chebyshev_comparison.jl` | Side-by-side trajectory comparison of CRM, `S_{μ*}` relaxed-MAP, and Chebyshev semi-iteration. |
| `numerical_experiments.jl` | Experiments of Section 8.5–8.6: iteration-count comparison of the five methods (Table 6), the empirical asymptotic rate of Strategy B, the residual histories of the convergence plot (Figure 3), and the parameter-free `C_T` on `V` versus parameter-tuned AAMR comparison on a `(θ_F, θ_p)` grid (Section 8.6, Table 8). |

Each script is self-contained and prints its results. Random seeds are
fixed where appropriate. `floor_analysis.jl` creates the `figures/`
directory on first run.

Theorem, proposition, section, table, and figure numbers in the script
comments and in this file refer to the cited version of the manuscript.

## Running

```bash
julia --project=. rate_verification.jl
julia --project=. worst_case_ray.jl
julia --project=. floor_analysis.jl
julia --project=. sharp_instance.jl
julia --project=. R30_experiment.jl
julia --project=. chebyshev_comparison.jl
julia --project=. numerical_experiments.jl
```

## What is verified

- The sharp rate ρ_V is attained on the explicit worst-case ray to ~1e-15.
- No sampled ray in *V* exceeds ρ_V (the bound holds).
- On *V*, CRM coincides with the optimal relaxed-projection step S_{μ*}.
- ρ_V < c_F² for θ_F < π/2, with one-step convergence iff θ_F = θ_p.
- Chebyshev semi-iteration beats ρ_V by a factor at most 2.

## Citation

```bibtex
@article{BelloCruz2026CRMrate,
  author        = {Bello-Cruz, Yunier},
  title         = {On the sharp linear convergence rate of the
                   circumcentered-reflection method on subspaces},
  year          = {2026},
  eprint        = {2606.07888},
  archivePrefix = {arXiv},
  primaryClass  = {math.OC}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Author

Yunier Bello-Cruz, Department of Mathematical Sciences, Northern Illinois
University — <yunierbello@niu.edu> —
[ORCID 0000-0002-7877-5688](https://orcid.org/0000-0002-7877-5688)
