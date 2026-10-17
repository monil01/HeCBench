# zerocopy: scientific and implementation observation

Zerocopysourceintentisamemory-placementexperiment,notjustvectoradd. GenuineportsmustretainbothactualhostallocationpathsandGPUaccessmappedCPUstorage;historicalprintedSUCCESS/zero-timesarenoevidence. FulloutputchecksandHostmemorytypeassertionsconfirmthenamedexperiment. Allocation/binding/freetimesareseparatefromtheprimarykernelregion;NUMA/interconnectandlargesizesweepnotprofiled.

## Reference audit

HistoricalJulia prints0timings/SUCCESSwithoutCUDAallocation,kerneloroutput. GenuineS1repair addsactualCUDAmappedhostalloc/register,devicepointerbinding,genuinevectoradd,andfulloutputs.

## Numerical and performance evidence

Configuration `zerocopy_both_host_mapping_modes_n1048579_tail_exact_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T195040175365Z_zerocopy/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

Reference one device candidate in two checks: initial shared B3 CuPtr host-constructor failure before device JIT then corrected integer-address constructor. KA and JACC each first device candidate/check. Historical placeholder repair separate from runtime/compiler feedback. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Observed prospective usage intervals are recorded separately; exclusive billed translation cost is unknown. Final-source proxies remain partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/zerocopy-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.


## Memory placement and numerical witness

All 2,097,158 values per trial match the independent dyadic integer-numerator oracle in all 20 artifacts. Both cudaHostAllocMapped and aligned malloc/cudaHostRegisterMapped paths assert actual Host pointer type after device-pointer binding. GPU kernels read and write CPU-resident mapped storage directly; no staging copies replace zero-copy. Native size sweep remains unmeasured beyond this named length with a partial final workgroup. Secondary allocation/binding/free samples are in study_host_stage_times.json; Julia binding includes wrapper work, so those are not matched API-only timings. Cache/load policy, alias metadata, NUMA and interconnect are unprofiled; observed timing differences have no established causal explanation.


Prospective rawusage intervals include cached/repeatedcontext and coordination; the parentworkflow contains its childphases and must not be summed with them. Exclusive billed translation cost remains unknown; source-size proxies remain separately labeled.
