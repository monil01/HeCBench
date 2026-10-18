# Eighteen tensor–adjacency contractions

## Mathematics and independent evidence

The unchanged native kernel computes18index-constrained contractions of T[a,b,c,f] and A[d,e], including diagonal identifications; output layout [x,y,case,f]. Allcases gate on positive adjacency except case6, which uses the signed adjacency directly. The named N9/C18 fixture has33distinct signed dyadic tensor values and39positive/36negative/6zero adjacency entries, exercising all branches rather than the original all-ones input. All26244outputs are checked in both original Float32/Float64 modes. Independent NumPy2.5.4 einsum expressions (listed in fixture_provenance.json) formulate the mathematical contractions without transcribing the kernel loop nests. Products are multiples1/128; absolute running sums≤546.75, scaled integers<2^24, so all intermediates are exactly representable in both precisions. Exact zero tolerance is declared before measurements; this does not characterize arbitrary rounding/conditioning behavior.

All40saved model/trial captures exactly match every oracle output, actual13207input/parameterprefix values and64guards. Lossless Float64 export is outside ROI, while actual device allocation/arithmetic remainsFloat32 orFloat64. No missing-output normalization, tolerance relaxation or selected fastest trial. `audit_saved_outputs.py` also verifies64explicit source/fixture bindings; all new model specs automatically bind every declared dependency including .bin inputs. Essential CPU-generated fixtures: study_tensor_f64.bin, study_adj_f64.bin, oracle_f64.bin with deterministic generator and hashes. Actual GPUcaptures staylocal.

## Scope and source fidelity

NamedN9 tests the complete18case/18channel operator, not original MakefileN64 performance. No application model, graph prediction quality or physical accuracy claim is inferred from source filenames. HistoricalJulia is only a smokemarker and fabricated zero timing/checksum printout; archived unchanged, not treated as numerical evidence. GenuineCUDAjl restores all18native cases/looporders/gates; native source unchanged, no corrected-native intervention. All three implementations remain framework_device_kernel, one independent outputworker with serial contracted-axis reduction, no communication/shared memory. Adjacent channels/cases share warps with differing branch/loop costs; no measured bottleneck attribution. KA and JACC retain the same puremath helper, ordinaryglobalindex/parallel_for; JACC explicitly sets shmem_size0. Native/Julia indexcompiler and launchcovariates remain, not abstraction-only causal estimates.

## Compiler-loop and usage

Each model uses one generic authored devicekernel/candidate, two actual precision-specific JIT process evaluations; all pass on firstcandidate, no compiler/runtime/numerical-feedback repair. Currentreference semantic restoration of a historicalstub is separately costed; 18mathematicalcases and five repetitions are not translation attempts. Two native process/build evaluations are precision cases, not algorithm revisions. Independent CPUgenerator preparation shared once; historical efforts unknown. Prospective cumulativeusage includes context/cache/coordination and shared windows overlap model phases; report separately from exclusive billed cost and savedsource proxies.

## Timings and reproducibility

Five checked in-process outertrials per model/dtype, full warmup before ROI, each ten unchanged kernel launches with blocking whole-device completion. Allocations/copies/checks outside ROI. Mean trials2–5. Actual configuration means in seconds:

[
  {
    "configuration_id": "contract_N9_C18_all18cases_signeddyadic_fp32_R10_v1",
    "precision_bits": 32,
    "result": ".porting-state/portable_julia/runs/20261018T045533075447Z_contract/result.json",
    "means_seconds": {
      "cuda": 7.868345e-05,
      "cuda_jl": 9.3211325e-05,
      "ka": 9.37616e-05,
      "jacc": 9.392885e-05
    },
    "scope": "N9C18/full18cases/26244outputs"
  },
  {
    "configuration_id": "contract_N9_C18_all18cases_signeddyadic_fp64_R10_v1",
    "precision_bits": 64,
    "result": ".porting-state/portable_julia/runs/20261018T045549651322Z_contract/result.json",
    "means_seconds": {
      "cuda": 9.924215e-05,
      "cuda_jl": 9.5911875e-05,
      "ka": 9.55669e-05,
      "jacc": 9.642685e-05
    },
    "scope": "N9C18/full18cases/26244outputs"
  }
]

Pinned Julia1.12.6/CUDAjl6.2.1/KA0.9.45/JACC1.4.0/localCUDA12.0,V100sm70; native -O3 sm70. `make run` verifies both source precisions using GPU6 serialized by study lock. Actual result/config/dependency hashes retained; no GPUretiming during metadata audit. Parent coordinator owns separate source/variant commits and manuscript integration.
