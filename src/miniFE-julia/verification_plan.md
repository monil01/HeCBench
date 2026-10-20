# miniFE: native readiness and prospective study plan

## Observed original execution

The archived original CUDA source builds with existing nvcc12.0 on GPU6. The original128³-cell run executes CPU Hex8 finite-element assembly and the full200-iteration GPU CG cap; its weak center-region checker passes. The original2³-cell run executes and reports center0.09375 versus its homogeneous analytic reference0.166667, failing the source0.06 threshold. These are original CLI diagnostics, not five-trial study evidence.

## Confirmed boundary defect and scientific reference

`boundary_witness.cpp/json` uses the unchanged original CPU assembly and boundary functions on27 nodes. The source defines eight nodes in both hot(x=1) and cold(other-face) sets. Eliminating the cold columns first irreversibly drops the later hot-edge contributions from interior RHS. Direct independent `b_free - A_free,bc*g` evaluation gives center RHS0.625; original ordering gives0.1249999998905, an error0.5000000001095. Removing the hot rows from the cold set before either elimination restores all RHS rows exactly. The source verifier itself explicitly prioritizes x=1 at the edges, supporting this minimal owned host-boundary correction. Original sources, witness and original executions remain preserved.

The selected equation is **-Δu=1**, not homogeneous Laplace: original `Hex8::sourceVector` has Q=1. Its analytic helper describes only the harmonic hot-face part. A prospective independent Poisson/Q1 residual check replaces that inapplicable center-only heuristic as the acceptance gate; the original heuristic verdict remains an auxiliary observation. Other CLI text/YAML time bytes are outside agreement.

The original two-point quadrature uses rounded Gauss coordinate0.577350269. The independent tensor-product matrix oracle must retain that source precision: normalized 1D mass diagonal `(1+g²)/2`, neighbor `(1-g²)/4`; stiffness diagonal2, neighbor-1. The assembled Q1 stiffness is the three tensor sums. The interior load is h³. This is not a hidden replacement with an ideal higher-precision quadrature.

The prospective native correction is conditional: `native_reference_repaired:true`, canonical hash-bound `reference_provenance`, explicitly excluded from untouched-native scientific/performance aggregates. No GPU mathematical body is changed. Do not start CUDA.jl until this corrected native study passes.

## Full primary computational scope

Original128³ cells /129³=2,146,689 nodes, full CPU assembly/boundary application and complete source CG200 cap/tolerance; small2³ and16³ controls proposed. Preserve the11 original kernel templates and actual selected path (seven called kernel types expected; count from execution, not this expectation). Original source allocation, copies, two-stage256-thread dot reduction and final state are retained. No fixed-step or one-matvec substitute.

Prospective contracts, to be finalized before measurement:

- Exact full actual initial CSR/input bytes, independently checked against the source-rounded Q1 tensor operator and correct Dirichlet RHS.
- Full returned solution plus final CG workspace vectors and true CSR residual; source convergence metadata and full partial-dot observations where already host-synchronized. No claim that every large transient vector at every kernel launch is archived.
- Independent true residual computed with long-double accumulation; proposed normalized L2 budget1e-7. Full finite values, boundary/max-principle controls. Continuum Poisson observations use both harmonic boundary and unit-forcing terms, with finite-element discretization error explicit.
- Declare exact/tolerance artifact budgets before the first corrected study series. Identical input/precision/regions and five checked trials per model, mean2–5.

## Input retention and disk constraint

Home currently has about12GB free. One full CSR readback is about0.7GB; repeating it inside every flat artifact would consume over20GB. Preserve a content-addressed **actual native device readback** once per case during readiness, with complete bytes and SHA256. Every measured model/trial must actually read back the complete CSR and all initial vectors and verify exact bytes against that bound retained readback. Record per-trial input provenance and retain the canonical bytes. A smaller flat exact prefix can include complete b/x0, with complete final state vectors in every per-trial artifact. This does not substitute hashes for an absent original input: actual canonical input bytes remain available. No deletion of earlier research, symlink relocation or OS/package changes is authorized/needed.

No target translation has begun. Native-readiness usage window99f5e4ebb3d1b274b0c27d24 remains open.
