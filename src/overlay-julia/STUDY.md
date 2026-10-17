# Overlay: full source detection-window RGB visualization

This is an image-visualization/compositing primitive, not detector inference or an application accuracy dataset. It blends one fixed source color into rectangular image regions. The source case640×480, srand123 pixel/box generation and245,760 boxes are all retained. The historical Julia program touched256 floats and printed PASS; it is preserved and replaced with the actual blend and full ordered launch loop.

## Source semantics and independent accuracy

Each rectangle reads the original input image and writes a distinct output array. All rectangles share color(255,204,203,1), with alpha=color.w/255=1/255, not normalized opacity1. Overlapping rectangles therefore write the same single-blend value, not repeated accumulation. Integer coordinates are nonnegative in the source fixture, with upper-edge image clipping; negative coordinates and alternate colors/opacity conventions are not certified.

Exact source C++ RNG draw order generates307,200 Float32 RGB pixels and245,760 four-Int32 boxes. Julia RGB NTuple3Float32 uses the same12-byte logical pixel layout; immutable boxes use16bytes. The independent CPU oracle uses a two-dimensional rectangle difference array to compute coverage, then Float64 affine blending with the actual rounded Float32 alpha/ialpha. This shortcut is CPU validation only. Every GPU implementation retains all245,760 source launches and all rectangle pixel work, with no union/fusion/subsampling optimization.

The fixture has307,198 covered and two uncovered pixels, maximum17,019 overlaps. All921,600 RGB output values are checked in every model/trial, with all1,904,647 image/box/color/shape prefix values compared byte-exactly. Box numbers are exactly represented in metadata; actual geometry remains Int32. A separate CPU audit checks all20 full artifacts, finite RGB/range values and exact unchanged pixels. Maximum Float64-oracle absolute error is7.62892886996e-6 for every model, below original1e-3. This tolerated rounding is an N3 observation, not a failed attempt or claim of exact real arithmetic.

## Actual work, mapping and timing

Each rectangle uses8×8 blocks and its original ceil grid, including inactive guards and image clipping. Source-level work is3,657,418,592 valid pixel updates and4,218,004,544 launched lanes per batch. The corresponding87,778,046,208 logical image read/write bytes are not measured DRAM/cache traffic. The independent union property must not be used to pretend the timed GPU only touches each covered pixel once.

CUDA.jl uses explicit CUDA launches, KA uses its2D global-index/local-group launcher, and JACC uses a2D callback with explicit zero shared memory. JACC's cached launch-specification block count is updated on every varied rectangle extent. These are own-framework device kernels with a shared pure Julia pixel helper; CUDA host allocation/completion does not establish cross-vendor portability. Vector loads/stores, FMA lowering, register/spill/cache behavior and host/runtime overhead remain unprofiled.

The timed region matches the source's whole245,760-iteration host launch/configuration loop plus device completion. It is not isolated GPU timestamp duration. One complete batch warms each process before five in-process full-batch trials; the final four are averaged, with no outlier selection. Resets/copies/transfers/oracle/serialization are excluded. Every Julia boundary uses blocking device-wide completion. The original case is feasible: a separate cold native diagnostic took about0.721s and passed the full oracle; it is not substituted for primary timing.

| CUDA (s) | CUDA.jl (s) | KA (s) | JACC (s) |
|---:|---:|---:|---:|
| 0.723114 | 0.810171 | 1.713810 | 3.275275 |

All required samples and dispersion are retained. The gaps occur with matched source work but include differing host dispatch/configuration and generated kernels. They are not pure device throughput or established launch-bottleneck measurements; profiling is required before causal attribution. Ordinary per-kernel roofline positions cannot simply be inferred from whole-batch runtime and nominal byte/FLOP counts.

## Effort and failure scope

The actual reference restoration and each portable target passed one candidate and one device-JIT check with one generic kernel definition and no compiler/runtime/numerical repairs. Shared pixel-helper/ABI/fixture/oracle/reference work is separately costed from target index/launch wrappers. Earlier framework knowledge informed the first drafts, including JACC mutable launch-grid handling; these are not context-free zero-shot success claims. Historical translation effort and internal compiler/specialization counts remain unknown. Consumed windows include cached context/reasoning/coordination and overlap broad/child phases, so they are not additive exclusive billed translation costs.
