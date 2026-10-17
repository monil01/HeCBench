# NMS: source pairwise detection-window suppression

The source README describes face-detector window postprocessing and lists a sample image/detection file that is absent in this checkout. The study uses synthetic boxes, not an image, trained detector, precision/recall experiment or application accuracy dataset. The source operator compares square-window intersection divided by the higher-score window area (with pixel +1 convention), not IoU. Strictly higher scores can suppress lower scores even when the higher box is itself suppressed; equal-score ties do not suppress one another. The verified mathematical target is this source operator, not a score-sorted greedy alternative.

## Full numerical contract

The fixture uses all4096 supported detections:512 separated clusters of8 positive-width square windows, widths8/12/16 and four dyadic score levels with ties. Inputs occupy65536 bytes. An independent Float64 all-pairs oracle uses the exact promoted source Float32 threshold0.3; the minimum active threshold margin is0.008641963, so this fixture does not probe arbitrary rounding-boundary cases. All input values are finite.

Every saved trial checks all16,777,216 pair bytes and all4096 keep flags, plus the exact input-byte prefix. Logical outputs have zero tolerance. The independent rowwise AND yields1024 kept windows. A CPU-only audit verifies all40 saved artifacts byte-exactly against the full oracle, confirms row invariants and verifies the existing CUDA.jl kernel bodies are byte-preserved. No original text parsing or real-detector accuracy claim is made.

## Native and per-model mapping

Native kernels remain unchanged: the map uses float4 AoS inputs,16×16 blocks and256×256 grid; the reduction uses128 lanes per row and32 collective AND rounds with global-byte updates. Every native lane stores the same vote result byte. This static concurrent-writer pattern is disclosed without inventing an observed sanitizer race; the named outputs all match the independent oracle. No native correction is hidden.

The existing genuine CUDA.jl reference uses SoA arrays, folds32 row partitions per lane, then performs one128-lane shared-memory AND tree and a single output writer. KA and JACC translate this associative mapping with their own index/shared/barrier APIs. All are framework device kernels; CUDA host allocation/completion does not establish cross-vendor portability. Native-to-Julia layout and reduction strategy differences are explicit performance confounds, not frontend overhead measurements.

CUDA.jl and KA use128-byte static/local scratch overwritten before use. JACC uses an explicit128-byte dynamic shared vector, initialized from a zero template with an extra barrier. Its map explicitly requests zero shared memory. All preserve the selected block/grid geometry. Cache behavior, physical memory traffic, registers, spilling, occupancy and compiler lowering are unprofiled. Source comparisons/division/logical operations need an explicit convention before assigning a roofline position.

## Separate original timer regions

Map and reduction remain separate named configurations. Each trial repeats only the selected kernel ten times and completes the device; the other stage runs outside that timer to prepare or validate the full pipeline. Resets/transfers/CPU checking/serialization are excluded. Warmup reaches both kernels before any timed trial. All Julia boundaries use blocking device-wide synchronization, matching native cudaDeviceSynchronize. Five trials are retained and the last four averaged. Repetition changes from the original Makefile1000 to named-study10; all input extent/stages are retained.

| Timed region | CUDA (us) | CUDA.jl (us) | KA (us) | JACC (us) |
|---|---:|---:|---:|---:|
| map | 166.458 | 272.662 | 363.824 | 365.294 |
| reduce | 344.318 | 42.009 | 43.690 | 153.445 |

The native-versus-Julia reduction gap includes a change from32 collective rounds to a local fold plus one tree. The additional JACC-versus-KA reduction gap remains unexplained without profiles; template staging alone is not established as its cause. All required samples and dispersion are retained.

## Effort, failures and usage

Both portable targets passed one candidate and two actual device-JIT check commands with two kernel definitions and no repairs. These are two timer configurations over one fixture, not extra translation attempts per numerical value or trial. Existing reference device bodies were not rewritten. The current adapter needed two program candidates: a host closure named map shadowed Base.map used inside CUDA's launch macro, causing a pre-JIT MethodError. Renaming host closures fixed it. One frontend rejection plus two successful JIT checks are three process evaluations, not three device compilations. Historical translation cost and internal compiler/specialization counts remain unknown.

NMS source inspection was interrupted for an urgent, separately metered MLP completion-boundary correction. Initial NMS windows closed, then resumed windows began before any NMS draft. Both nonoverlapping inspection/preparation segments are retained. A resumed workflow count is not a complete single-window benchmark cost. Context/cache/reasoning/coordination are included, broad and child windows overlap, and exclusive billing is unknown.
