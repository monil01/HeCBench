# present: scientific and implementation observation

GenuinePRESENTportsmustproduceactualciphertext, unlikehistoricaltouchkernel. TheyretainnativefusedbyteSbox/permutationtablesandbytekeyschedule, whileindependentPythonuses80bitintegerrotationandbitpermutation. Allfourpublishedknownanswersareembeddedinthenonuniform1025blockfixture, andeveryciphertextbyteischecked. Thisestablishesconfiguredalgorithmconformance, notsecuritycertification/wholekeyspacecoverage. Primaryknownanswersource: https://crypto.orange-labs.fr/papers/ches2007-450.pdf

## Reference audit

HistoricalJulia writessequentialbytes andhardcodesPASS, notPRESENTencryption. GenuineS1repair retainsnative31roundbytefusedSbox/permutationtables,keyrotate/counterinjection,private8/10bytestateandoneworkitem/cipherblock.

## Numerical and performance evidence

Configuration `present80_n1025_full31_four_published_kats_nonuniform_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T193755183367Z_present/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

CUDA.jl reference: two device candidates and three process checks: initial shared fixture host B2 parse failure before JIT, then actual B4 InvalidIR, then verified stable argument-helper repair. KA/JACC counts are recorded independently. Historical touchstub S1 review separate from observed compiler attempts. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Observed prospective usage intervals are recorded separately; exclusive billed translation cost is unknown. Final-source proxies remain partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/present-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.

## Independent specification witness

All 8,200 ciphertext bytes in each of all 20 model/trial artifacts match a separate Python specification using 80-bit integer key rotation, nibble substitution, and bit permutation. The first four fixture cases match the published answers in Appendix I, physical PDF page 16, of the [original PRESENT paper](https://crypto.orange-labs.fr/papers/ches2007-450.pdf). This checks configured PRESENT-80 algorithm conformance, without establishing security certification or exhaustive key-space coverage. The native byte-fused table algorithm is retained, while the oracle uses bit operations.

JACC initially used an incorrect host launch API and failed before device JIT; the archived source and B3 event remain recorded. Corrected launch_spec and @init_backend then compiled the first device candidate.


Prospective rawusage intervals include cached/repeatedcontext and coordination; the parentworkflow contains its childphases and must not be summed with them. Exclusive billed translation cost remains unknown; source-size proxies remain separately labeled.


Usage attribution scope: PRESENT shared benchmark_workflow/reference_preparation intervals include a brief unrelated stencil3d metadata scope adjudication. They are not exclusively attributable to PRESENT. KA and JACC translation windows contain no other benchmark work.
