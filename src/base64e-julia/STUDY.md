# base64e: scientific and implementation observation

Base64 encoding is exact byte manipulation coupled to a transfer/stream experiment. Genuine versions compute allpaddingclasses andretainbothonequeue/fourstreamtransferpipelines. HostCUDAinterop isexplicit, andthematching256groupchunksavoiduntestednativeoverlapfromroundedlaunches. Completeoutputbytesarecheckedindependently. These3KiBfixturesdo notestablishlarge-messageoverlaporbandwidthscaling, andrequiredmean-last-fourreplacesnativebest-ofcycleaggregation.

## Reference audit

HistoricalJuliaonlycopiesrepeatedinputbytes andhardcodesPASS, neitherBase64bits/padding norstreams/transfers. Genuine reference now implements3bytesto4alphabetbytes andmatchedsingle/fourstreamtransfersequences.

## Numerical and performance evidence

Configuration `base64_padding012_n3072groups1024_one_four_streams_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T183527226074Z_base64e/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

CUDA.jl firstprocess stoppedbeforedeviceJIT due sharedstreamFFItype namespace; oneactualdevicecandidate,twoprocesschecks,onesharedhostadapterrepair. KA/JACCcounts recordedindependently. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Observed prospective usage intervals are recorded separately; exclusive billed translation cost is unknown. Final-source proxies remain partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/base64e-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.


Independent Python standard-library Base64 checked everybyte ofall20savedartifacts and decoded each ofthe6expectedcaseoutputs backtooriginalinput. NativeCPU24bitencoder andJuliaBase64stdlib checks also pass. Exactly256groups/fourstreamchunk makeslaunchchunksdisjoint; this doesnotcertifytheoriginalroundedchunksforarbitrarysizes or prove actualtransferoverlap.


Prospective rawusage intervals include cached/repeatedcontext and coordination; the parentworkflow contains its childphases and must not be summed with them. Exclusive billed translation cost remains unknown; source-size proxies remain separately labeled.
