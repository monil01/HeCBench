# HACC short-force study

## Scientific provenance

The repository links this kernel to LLNL CORAL HACCmk. The [official LLNL summary](https://asc.llnl.gov/sites/asc/files/2020-06/HACCmk_Summary_v1.0.pdf) identifies a cosmological collisionless N-body short-force microkernel. This study tests the extracted force operation, not a full cosmological simulation. The old generated ontology's graph-analytics classification is contradicted by source and primary provenance; the reviewed study layer records particle-force cosmology.

The operation sums mass-weighted Cartesian displacements times a softened inverse-cube radial factor minus a fifth-degree polynomial, masked at squared distance0.5, then scales the additive output update by0.23. All arithmetic/storage is Float32. Coordinates and masses are structure-of-arrays, with independent target ownership and no atomics or shared memory.

## Reproducible original-size inputs

The original784 targets and15000 source candidates are retained:11760000 logical pair iterations in four256-thread blocks. The240000-byte fixture reproduces the source's sequential Float32 coordinate/mass recurrence. A checked-in generator/hash record and a native C++ every-byte recurrence comparison establish shared inputs. These synthetic points lie approximately on a line; no real cosmological snapshot or physical units are supplied.

Every trial checks all2352 active Cartesian components. Original unused host tail zeros are not counted as GPU outputs. Native unchangedFloat32gold and full dual artifacts retain absolute tolerance1. Independent Float64 radial mathematics also requires absolute1 and relative1e-4. Measured independent maximum absolute error is0.14179583 and relative error3.55811e-6. Native versus Julia maximum difference is0.01171875; all three Julia models agree exactly. All values must be finite. No trajectory or global conservation claim is made for the subset of targets.

## Timing and interpretation

One source-equivalent kernel update per trial, resets/copies outside timing, explicit JIT warmup, five synchronized in-process trials, mean runs2–5. Original four-block geometry and cache-resident inputs are preserved. This is a small-grid microkernel, not full-device scaling evidence. Square root/division and polynomial arithmetic require a mixed-operation performance model; no measured bandwidth, register, occupancy or compiler cause is asserted.

The genuine CUDA.jl device-kernel text is unchanged; its full original program is archived and only an entrypoint guard/study adapter added. KA uses an outer bounds guard and JACC requests exactly zero shared bytes. Each target passes its first candidate/check. Native adapter instrumentation needed a second build after a duplicate diagnostic variable name; that shared build error and repair are separate from target translation. No mathematical native or Julia reference repair was needed. A diagnostic reader's missing-runlog assumption is separately recorded as shared tooling effort.

Prospective phase usage includes cached context, provenance/fixture work and coordination. Partial saved-source proxies remain separate; broad workflow overlaps phases and is not additive or monetary billing. Historical Julia generation effort is unknown, and JACC intentionally starts with successful KA mapping available.
