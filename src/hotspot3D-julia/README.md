# hotspot3D — time-level fidelity in a thermal update

RQ1: Original source models chip temperature with power and ambient exchange using64x4 column threads and serial z rolling registers. Four named cases retain64/128/512 grids and2/3/5/8 layers over1/2/3/5 steps, including the original512×512×8 footprint but not5000 steps. All20 full15,400,493-value captures pass independent Float64 whole3D stencil checks (maxabs 6.78579350506e-05) plus exact input, guards and readonly power; maximum actual last-step temperature-sum-balance residual 2.07207627906e-07. This is a discrete thermal consistency check, not complete physical-model validation.

The source public even-step selector returns the previous buffer. Its CPU helper has matching output-buffer parity, so original RMS agreement hides this. On the2-step64×64×3 case,12,280 public values differ from actual final values by up to0.0228424072265625. Both actual final and previous arrays are recorded and independently checked. The primary is explicitly known-native-defect reproduction and excluded from untouched-native scientific/performance aggregates; no corrected-final secondary is claimed.

RQ2: One CUDAjl device body passes after a shared host comprehension-parse repair (two complete program evaluations, one device check). KA and JACC each pass their first source/check after shared helper/oracle reuse. A native-readiness asset generator parse failure caused a missing-file native exit before GPU launch; this is owned T1 infrastructure, not unsupported CUDA. An unavailable optional NumPy dependency led to standalone C++ fixture/oracle generation without changing shared environments. Real input/cached/output/reasoning phase counters are preserved; repeated cached inherited context, verification waits and coordination mean these are marginal workflow activity, not exclusive or billed translation cost. Native readiness, shared verification and annotation windows are separate; verification is left censored for initial submission and overlapping windows must not be summed.

RQ3: Means trials2–5 in microseconds: cuda 319.66100000, cuda_jl 352.77075000, ka 491.74700000, jacc 596.67275000. ROI sums original kernel loops, pointer swaps and context completion across four cases. Actual allocation, transfer, full before/after readbacks and frees execute outsideROI. These are not per-size throughput or original full-CLI offloading times; no outlier removal or retiming occurred.

RQ4/RQ5: All three use own framework device indexing/dispatch and shared pure scalar math, no CUDA-specific device extension. Matching geometry/z algorithm reduces one confound, but dispatch and native fusion still differ. No actual instruction/resource/traffic counters were collected. Tested API expression is demonstrated; portability/performance or scientific-final-output recommendations are not established by reproducing this defect.

Reproduction: run the frozen `study_config.json` through `tools/verify_portable_julia.py`; the runner owns the GPU6 lock. `audit_outputs.py RESULT` rehashes and checks captured arrays on CPU only. Original sources/Makefile remain byte-preserved in reference_history. New code has no dependency on NumPy.

## Clean-checkout data regeneration

The large fixture/oracle binaries are generated, not required Git assets. From the repository root:

```sh
cd src/hotspot3D-julia
g++ -O2 -std=c++17 generate_fixture.cpp -o /tmp/hotspot3D_fixture
/tmp/hotspot3D_fixture
sha256sum fixture.bin oracle.bin fixture_manifest.json
```

Expected hashes and byte counts (recorded compiler g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0):

- `fixture.bin`: `9057484e27b4e3be70ff2e587a37d8ce693772dbc1787608a2946147d5732d5f`, 26400948 bytes.
- `oracle.bin`: `d973feb19a8e3cd02a81e0453c37a3b2381e050c7257a2e7c85daa1a9ab3d916`, 70402048 bytes.
- `fixture_manifest.json`: `e8dc0067d0902d84c8a09f7360d50582045844a680ead2b032f63a43dae09200`, 1719 bytes.
