# wordcount: scientific and implementation observation

Wordcount provides a source-policy and evidence-strength example: the native predicate includes six punctuation bytes between ASCII uppercase and lowercase ranges. An explicit isolated-punctuation fixture gives sourcecount6 but ordinaryASCIIlettercount0; the all256byte fixture gives sourcecount1 versus ordinarylettercount2. All four versions preserve the actual finite byte-set policy, rather than silently repairing it. Independent maximal-run grouping checks complete scalar outputs on seven edge and long fixtures. No linguistic or per-byte flag correctness claim follows from scalar-only native output.

## Reference audit

Historical CUDA.jl only writes a one-element marker and prints hardcoded wordcount/PASS/time lines without processing text. Preserve those bytes; new reference computes actual source byte-run counts. Native is_word_start/is_alpha functors are unchanged; named adapter retains same Thrust inner_product expression on borrowed preallocated input, excluding original CLI allocation/H2D/free for every model.

## Numerical and performance evidence

Configuration `wordcount_source_byteinterval_preallocated_seven_edges_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261018T011127651328Z_wordcount/result.json`. Five in-process trials per model; all seven native scalar outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native byte predicate and Thrust inner-product reduction semantics preserved in a named preallocated adapter. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

Historical CUDA.jl smokestub S1 repaired in separate shared reference phase. One device candidate/program check per Julia model, no observed target compiler/runtime/numerical repair. All3author one actual predicate/reduction definition; native remains source-functor Thrust. Prospective activity windows include metadata/verification, not exclusive coding cost. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Observed prospective usage intervals are recorded separately; exclusive billed translation cost is unknown. Final-source proxies remain partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/wordcount-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.

## Source-policy and evidence strength

Native predicate65<=byte<=122 includes bytes91..96. Six isolated punctuation characters give nativecount6 but ordinaryASCIIlettercount0; the all256byte fixture gives sourcecount1 versus ordinarylettercount2. This study preserves the native byte-set policy and makes no linguistic/Unicode tokenization claim. Independent maximal-run grouping confirms all7scalar counts in all20captures: [0,1,0,6,1,14848,373]. Evidence strength is `aggregate_count_only`; there is no pointwise flag artifact or claim that per-byte errors cannot cancel.

All actual device input bytes are copied back before input/control prefix serialization. Native uses unchanged source functors and the same Thrust200001 inner_product expression on preallocated borrowed input. Julia models have genuine authored128thread predicate/shared-tree/global-atomic kernels, not a renamed native wrapper. This is a documented library-to-custom-algorithm adaptation; native internal geometry is unprofiled.

The named preallocated ROI includes reset/temporary/reduction/scalarcompletion; original source device-vector allocation/H2D/free is excluded for all4 and remains unmeasured. Julia scalar extraction allocates a host array; native reduction/front proxy have their own scalar transfer paths. `per_component_performance.json` retains7component×5series and means trials2–5. All3 use512Bshared and8algorithm barriers; JACC additionally initializes its entire template with one initialbarrier. No large-stream/overflow or causal bottleneck claim. Usage windows include verification and metadata activity.


Prospective rawusage intervals include cached/repeatedcontext and coordination; the parentworkflow contains its childphases and must not be summed with them. Exclusive billed translation cost remains unknown; source-size proxies remain separately labeled.
