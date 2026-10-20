# Independent XLQC mathematical review

Scope: read-only original-source review and separate CPU-only diagnostics. No GPU execution, target draft, original source edit, system package change or SCF acceptance claim. Results are under this private directory, outside the owner's source tree. Usage is a distinct `xlqc/shared/independent_oracle_review` window on the portable_infrastructure stream.

## Completed evidence

- `clone_provenance.json`, `cints_tight.patch`, `prepare_tight_oracle.py` preserve exact original CPU source identity and a **CPU oracle only** qualification: EPS1e-9→1e-14, ITMAX100→1000, exact zero-limit/convergent integral series for Boys arguments≤1 replacing the original1e-11 clamp. No native CPU or GPU implementation was repaired.
- Original Boys versus independent64/128-point integral definition: sampled orders0…4, arguments0…300, maxabsolute1.5158430067e-11 and maxrelative1.2110101624e-10; quadrature64/128 difference≤3.62211e-15. These are floating diagnostics, not a certified interval bound.
- `full_primitive_cpu_review.json`: all194,481 finite H2O/STO3G primitive quartet values compared original versus tight CPU oracle. Maxabsolute3.2643956982e-12; tight Boys versus independent128-point quadrature≤2.96291e-15; quartet symmetry errors≤2.77556e-17. Full CPU calculations each took about0.2s.
- Packed21×8 primitive data was independently reconstructed from the recovered exact basis/geometry, using source angular order, Bohr conversion0.5291772086 and Cartesian normalization formula. **Owner must verify this reconstructed table against the actual native packed device-input readback before adopting it.** It is not yet GPU-validated input evidence.
- `contraction_index_review.json`: full primitive and contracted J/K source indexing agreed with separately expressed dense four-index tensor contractions. Max primitive errors2.22e-16, contractedJ1.33e-15 andK3.33e-16. Control density0.1I+0.01outer(1…7) is a nonzero symmetric SPD algebra control, not an actual SCF state. Original-vs-tight primitive J/K uncertainty on this density≤7.65e-12/2.47e-12.
- Tight Float64 ERIs with **SP-rounded primitive inputs** versus original DP inputs differ by≤1.84816e-8. This separates input rounding only; it does **not** simulate all single-precision Rys arithmetic or certify a GPU SP error budget.

## Correct full primitive contractions

Let g_i be the contracted-basis index of primitive i, and T_ijkl include each packed coefficient×normalization once. For each actual uploaded density (D_prev initially, D_diff subsequently):

- J^P_ij = Σ_kl T_ijkl D[g_k,g_l]. Source triangular k≥l loop doubles **primitive** k≠l only, even when g_k=g_l.
- K^P_ik = Σ_jl T_ijkl D[g_j,g_l], with the full j,l domain and no triangular doubling.
- Contracted J_ab andK_ab sum all ordered primitive pairs with g_i=a,g_j=b. For a=b, both primitive orientations participate. Do not introduce another factor2 for contracted off-diagonal indices.
- Validate all231 packed primitive J andK values, not only the7×7 contracted matrices or final energy; contraction can hide compensating primitive errors.
- For a source-faithful screened comparison, bind actual uploaded Q, D and masks; compare separately against the unscreened physical tensor contractions, retaining omitted term magnitudes.

### Screening caveat

`main.cu:354` calls Q a square-root Schwarz quantity, but line357 assigns `calc_int_eri_rys(a,b,a,b)` directly, and `basis.cc` returns the raw contracted integral. No square root is taken. Preserve actual raw-Q screening semantics rather than correcting the comment into the computation. Moreover a contracted-pair quantity is not automatically a rigorous bound for every coefficient-weighted primitive term. The independent unscreened gate can measure the error in the declared small molecule; it does not establish a general rigorous screening bound.

## SCF invariants and state association

Source `Coef_to_Dens` uses D=2 C_occ C_occᵀ. For closed-shell H2O with5 occupied orbitals:

- Cᵀ S C=I; occupied projector gauge/sign does not matter.
- Tr(D S)=10; D S D=2D, **not** D S D=D.
- Whitened density S^(1/2) D S^(1/2) has eigenvalues0 or2 and is positive semidefinite within numerical error.
- For the actual matrix diagonalized, F C=S C diag(orbital energies). Associate this residual with the **post-DIIS** Fock, not an independently recomputed physical Fock of a different density.
- Source incremental Fock uses F_prev+J(D_diff)-0.5K(D_diff). F_prev is saved before DIIS. Validate that provenance and compare the accumulated physical Fock against a fresh H+J(D_current)-0.5K(D_current), with screening differences measured rather than assumed zero.
- Physical SCF residual F(D) D S-S D F(D) need only approach zero at convergence; require neither zero at arbitrary intermediate states nor monotonic energy under DIIS.
- Source reported energy uses newD with post-DIIS Fock. Preserve that exact source observable. Independently compute the physical HF functional0.5 ΣD_ab[H_ab+F(D)_ab]+E_nuc using **fresh unscreened F(D)**, especially for final convergence checks. These are distinct quantities before convergence.
- Record source convergence deltas (relative energy, RMSdensity, maxdensitychange), actual iteration count, and physical functional/residual checks separately. Near-equal scalar energies cannot substitute for full matrices.

## Integration recommendations and limits

Keep tight CPU oracle symbols separate from original cints/GSL controller symbols. Prefix **all** cints names in a standalone translation unit before including cloned source/header, then export a unique C ABI function; do not rely on coincidentally safe dynamic symbol interposition. Bind clone, exact diff, original source, compiler flags and helper hashes. Store DP-input and SP-rounded-input ERI caches separately, preserving native actual dtype/operation boundaries. The native full pipeline and numerical budgets remain the owning worker's independently predeclared contract; this review has not executed or certified any GPU/SCF trial. Use CPU numerical convergence/cross-formulation evidence to qualify oracle accuracy, not a post hoc tolerance chosen to accept a GPU result.
