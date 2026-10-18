# keccaktreehash: scientific and implementation observation

The historical tree-hash benchmark uses Keccak-f[800]: 25 lanes of 32 bits and 22 rounds. It is not the 1600-bit SHA-3 sponge. Its source layout is part of the contract: the CPU root absorbs contiguous eight-word chunks from the actual structure-of-arrays leaf buffer. Each measured pipeline includes H2D transfer, leaf hashing, D2H transfer, and CPU root work across eight passes whose root state carries forward. Consequently these are pipeline timings, not device-only hash throughput. Full leaf chaining words and every 25-lane root state are compared. An independent coordinate-form permutation with LFSR-generated constants is first checked against two published XKCP known answers, then used for complete measured-output verification. No assertion about standardized tree hashing, padding, domain separation, or cryptographic security follows from this numerical agreement.

## Reference audit

HistoricalCUDA.jl touches UInt32array withXOR thenprints hardcodednative-lookingdigest andPASS; noKeccakpermutation,absorption,leafhashorCPUroot computation. Preservestubbytes; replacewithgenuine22round32bitKeccakpipeline.

## Numerical and performance evidence

Configuration `keccakf800_tree64x3_leafblocks3_twoinputs_eightpasses_fullstates_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T235220392893Z_keccaktreehash/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

CUDA.jl reference needed two candidate validations: four wrongly transcribed round constants made its local CPU/GPU computations agree but disagree with native input prefixes and all outputs. Exact prefix checks rejected every trial. The corrected constants match both source and independently generated LFSR values. KA and JACC each start from validated reference data; target compiler repairs are recorded separately. The intentional external-witness rejection of the old candidate is verification evidence, not another compiler attempt. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Observed prospective usage intervals are recorded separately; exclusive billed translation cost is unknown. Final-source proxies remain partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/keccaktreehash-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.

## Independent primitive and complete pipeline witness

All 20 saved captures match an independent 5×5 coordinate permutation exactly. Its round constants are generated from an eight-bit LFSR, rather than copied from either implementation. The oracle passes both published full 800-bit permutation examples: zero input and the first output reused as input. See the [Keccak specification summary](https://keccak.team/keccak_specs_summary.html) and [XKCP Keccak-f[800] intermediate-value vectors](https://raw.githubusercontent.com/XKCP/XKCP/master/tests/TestVectors/KeccakF-800-IntermediateValues.txt). All 1,536 leaf chaining words and eight complete 25-lane root states per case are checked, across two fixtures.

The first new CUDA.jl reference mistranscribed four round constants. Its GPU and local CPU checks agreed because both used the same wrong table, but exact prefix comparison rejected all five trials; every output word differed from native. The failed configuration, source dependency, raw outputs, and independent witness rejection are preserved. One numerical feedback repair corrected those constants. This illustrates a common-mode local-check failure, not a defect in the native algorithm. KA and JACC each passed their first candidate validation after the reference data was corrected.

The measured native CPU/GPU sources are unchanged. A guarded-header macro configuration uses three thread blocks and three input blocks per leaf, retaining 64 threads and 22 rounds. Default dimensions, the unused two-stage CPU tree and stream cipher modes remain unmeasured. The actual SoA leaf layout and contiguous eight-word root absorption are retained. Each whole pipeline includes eight H2D/kernel/D2H/CPU-root passes and carries root state between them. Time is the sum for both fixtures, not divided by eight. Preallocated 200-word diagnostic root history is written inside every model's ROI. No device-only throughput or cryptographic-security conclusion is implied.


Prospective rawusage intervals include cached/repeatedcontext and coordination; the parentworkflow contains its childphases and must not be summed with them. Exclusive billed translation cost remains unknown; source-size proxies remain separately labeled.
