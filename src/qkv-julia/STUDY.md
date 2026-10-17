# qkv: scientific and implementation observation

Native default1 projects tokenfeatures using a naive one-output-per-thread FP32 affine matrix product. Historical selectors3/4 preserve the operator but change mapping; this study names default1 explicitly and does not claim coverage of the optional vendor or tiled variants. All24576outputs arechecked using an independent exact integer-dot witness. Results measure this primitive, not wholetransformer quality or attention.

## Reference audit

HistoricalJulia genuine affineprojection, columnmajorlayout adapted. Selector4aliasesnaive1 insteadnative sharedtiled4; selector3usesunfusedGEMM+biasinsteadoffusedcuBLASLt. TheseareP1mappingcompromises, notwrongscientificoperator. Namedstudymeasuresonlydefault1 withnative rowmajorlayout.

## Numerical and performance evidence

Configuration `qkv_default1_bt128_c64_oc192_dyadic_bias_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T182102305753Z_qkv/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

One saved candidate and one compiler/JIT process per target model; no observed target repair. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Observed prospective usage intervals are recorded separately; exclusive billed translation cost is unknown. Final-source proxies remain partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/qkv-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.


Independent witness: all20savedoutputarrays exactly match a separately implemented Python integer matrix product plusbias, converted by the known256denominator. Prospective rawusage intervals include cached/repeated context and coordination; the parentworkflow contains the childphases and must not be summed with them. The exclusive billed translation cost remains unknown. Native optional variants2/3/4 are an explicit unmeasured mapping backlog, not unsupported framework features.
