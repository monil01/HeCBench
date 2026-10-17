# bitpacking: scientific and implementation observation

MathematicalbitstreamagreementalonewouldnotmakehistoricalCPUconfiguration/unbufferedpackingaperformancematchedtranslation. GenuinecomparisonrestoresnativeGPUreductionandcooperativebuffering,thenchecksallconfigurationpartialsandpackedwordsratherthansampledunpackonly. Smalltailcasesand4096gridstridecasecoverdifferentmappingboundaries;zerowidth/otherdatatypesand100millionsizearenotcertified.

## Reference audit

HistoricalJuliaimplementscorrectbitpackingbutCPUmin/maxreplacesnative2GPUreductionsandperwordglobalreadsreplacecooperativesharedinputbuffering. P1mapping/perfregioncompromise,notwrongscientificoperator. Restorenative3GPUkernelpipelineforcomparison;preserveoldsource.

## Numerical and performance evidence

Configuration `bitpacking_int32_gpu_scan_finalize_buffered_pack_sixsizes_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T195725333419Z_bitpacking/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

Reference GPU mapping repair is separate from target translation: historical operator output is meaningful but host reductions/unbuffered packing do not match native mapping. A shared B2 host comprehension spacing error was repaired before device JIT; no compiler feedback repair is implied. Target attempt counts are independent. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Observed prospective usage intervals are recorded separately; exclusive billed translation cost is unknown. Final-source proxies remain partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/bitpacking-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.


## Full bitstream witness and mapping scope

Every packed word, zero padding word/bit, minimum, bit width, and scan min/max partial agrees in all 20 model/trial artifacts. A separate byte-window decoder reconstructs all 2,008,923 input values per artifact and checks their independent fixture formulas; it does not reproduce GPU shared-buffer packing. The largest named size crosses the 4096 scan-grid cap, while size2 checks the small reduction tail. Zero-width/constant inputs, other integer types, and native100000001size remain outside this measured contract. Historical CPU reductions/unbuffered packing are classified P1 mapping, not incorrect integer compression. KA uses GPU-only cpu=false kernels with documented synchronization helpers. JACC copies shared templates (2048B scan/finalize,1024B packing) and adds initialization barriers; this overhead is recorded separately from matched required shared capacity.


Prospective rawusage intervals include cached/repeatedcontext and coordination; the parentworkflow contains its childphases and must not be summed with them. Exclusive billed translation cost remains unknown; source-size proxies remain separately labeled.
