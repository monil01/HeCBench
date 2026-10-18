# MiniMod defined-forcing wave/PML component

This named conditional study advances a complete padded3D wavefield using all seven source interior/PML regions, point forcing and two-level swaps. It preserves the original native GPU bodies byte-for-byte in native_kernels.inc, extracted before the original host target wrapper. Original CUDA and genuine Julia CLI sources remain archived. Historical Julia used global-memory stencils and different hostFMAX/VMAX/CFL constants; the current reference restores source shared tiling and freezes common numerical inputs. No original fullCLI certificate is claimed.

## Input and source scope

The original host caller allocates nsteps forcing samples, while target consumes source[1] through source[nsteps], reading one past that allocation. The named input provides nsteps+1 initialized values, retains that exact source convention and gives unused source[0] a nonzero canary. This host allocation/initialization intervention is explicitly conditional and excluded from untouched-native aggregates. It is not a GPU algorithm correction, and no undefined-input GPU result is fabricated.

The grid is24³ with eight-cell PML slabs, source padded strides40×32×32, radius4 and8³ logical threads. All seven regions are nonempty, disjoint and aligned. The native default20-cell slabs have unaligned shared-load tails: exceeding an axis stride may still address a neighboring padded row and may only feed masked lanes. That static concern is not proof of allocation OOB, native numerical failure or framework limitation. It is recorded as unvalidated default shape scope, not an observed failure.

Three cases cover four steps of point forcing with variable(c*dt)^2 and nonzero damping/memory, two steps from a manufactured quadratic field with zero damping, and a zero solution with nonzero damping. Native source leading padding20floats is preserved so logical(0,0,ndampz) remains128B aligned; these20 leading words and eight added trailing words are guards for every field. All payload padding/ghost values and forcing samples are explicitly initialized.

## Numerical evidence

The independent CPU oracle uses direct global25-point eighth-order finite differences and Float64 wave/PML recurrences with actual Float32 inputs/coefficient values. It does not reuse the GPU shared-tile algorithm. Fixed budgets declared before GPU work are2e-6+2e-5relative for modified wave values and1e-10+2e-5relative for the much smaller phi field. Every actual input byte, immutablevp/eta field, untouched wave halo, phi outside PML and guard is exact; the complete zero case is exact. A CPU analytic quadratic-Laplacian residual7.65e-10 checks the oracle; it is not a separately captured GPU first-step experiment. GPU evidence covers complete final fields against that independently checked oracle and both references.

Each of20 artifacts contains1844076 Float64 words losslessly encoding actual Float32 values:614436 exact prefix words and1229640 full pre/post wavelevel/phi/vp/eta/guard fields. Per-field errors, normalized budgets and exact-mask counts are retained in output_audit.json. No checksum, sampled subset, output canonicalization or Float64 GPU computation is implied. The point-forcing case develops nonzero phi, so an implementation omitting that memory update cannot pass its separate budget.

## Mapping and timing

The original16³ Float32 shared tile occupies16384bytes per512-thread stencil block. It initializes two octants, synchronizes, loads radius-four faces, synchronizes again, then computes the source expression. CUDA.jl and KA retain physical8×8×8 threads. JACC flattens to8×64 threads and correspondingly flattens logical groupy/z, preserving logical coordinates and linear warp order. This avoids the pinned JACC shared-vector initializer ignoring threadz; no unsupported or unrun failure is claimed. JACC also copies all4096 template values and adds an entry barrier, extra work retained explicitly. The point-source kernel uses zero shared memory.

Each checked trial times the sum of three complete step components:56 stencil/source launches, with completion once per component as in the original target loop. Setup, resets, actual preexecution GPU input readback, warmup, final readback/oracle and serialization are outside. Five raw samples remain; only2–5 enter the mean. Small grid, restored reference tiling, flattening/template work, host launch setup and floating expression ordering are explicit covariates. These are not original default-CLI time-to-solution, peak throughput or measured roofline results.

Four-way numerical agreement checks the stated discrete operator. It does not establish physical PML reflection/absorption quality, resolution convergence, long-time stability or complete seismic application correctness. Current GPU candidates pass their first compiler/check invocation. A generator syntax repair occurred before fixtures or GPU compilation and is toolingT1, not a framework compile failure. Shared helper/oracle work belongs to the reference phase and is reused KA then JACC. Prospective windows include cached/repeated context and coordination, overlap the outer workflow, and are not additive or exclusive billed costs. Historical cost remains unknown. Root alone commits shared work and each variant separately.

### Achieved small-field accuracy

The pulse phi field has maximum magnitude1.04038e-10. Its predeclared1e-10 absolute budget is therefore weak relative to this very small field: replacing the entire field with zero would fail the largest point budget by only about4%. Actual recorded maximum phi error is1.52334e-18, or1.464e-8 of the oracle field maximum. This read-only achieved-accuracy analysis does not change acceptance or rerun GPU work. Do not describe the acceptance budget as a strong relative-resolution guarantee for every tiny phi value.

Pinned JACC2D parallel_for also calls device/max-grid attribute lookup functions on each stencil launch. These source-visible host calls fall inside the measured interval; their caching and latency contribution are unmeasured. They and template copying are covariates, not isolated explanations for the stable runtime difference.
