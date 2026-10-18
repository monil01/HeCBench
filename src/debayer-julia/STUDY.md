# debayer: scientific and implementation observation

The source reconstructs packed3-byteRGB into rowswith4wbytepitch; treating theallocation asRGBAwould change semantics. Exactinteger sparse-convolution oracle confirms everyRGBchannel andzero rowpadding, including truncation/saturation, allfourBayerphases, partialtiles andsourceboundarypolicy. Separate sample-preservation andconstantimage witnesses supplement dualreference agreement. The repeatedoutputreset belongs inside thetimedregion because originalsource timescudaMemset+kernel. Smallshapeone-foldreflection limits are documented without pretendingallCLIinputs work or claimingimagequality/calibration.

## Reference audit

HistoricalCUDAjlonlytouchesscratchandprintszeros, no Bayerinterpolation. Restore actualinteger MHCshared-apron32x5tile withall4patternsupport, actualRGB3bytepixel/4wrowpitch, truncatingresponses/saturation andsource-timedruntime memset+kernelrepeat. Nativeheadersunchanged.

## Numerical and performance evidence

Configuration `debayer_integer_MHC_all4patterns_5fixtures_RGB3pitch4w_memset_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261018T044205018030Z_debayer/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

Firstsavedcandidate andwholeprogram/JITprocess for eachJulia model. Shared source/reference/oracle preparation reused beforeKA thenJACC; ordered marginal work, not independentcoldstarts. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Prospective activitycounters retainedincludingcached/repeatedcontext andmarkerlag; notexclusivebilledcost. KAwindow includes separatedsharedmetadatahelperB2fix. Historicalusageunknown, sourceproxiespartial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/debayer-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.

## Complete image/operator evidence

All20artifacts bitidentical:2905exactinputprefix+11520fullRGB/paddingbytes+20rawguards=14445UInt32encodedwords. Independentrawbyteguard/prefix, all2880actualmosaicsamples, allrowpaddingzeros andconstant123RGB preservedineverycapture. Sharedapron1296B native/CUDAjl/KAstatic vsJACCdynamic1296B+324wordpublictemplatepreload. Ownframeworkdeviceentry each; CUDAhostruntime memset insideROI explicit, noforeigndeviceintrinsics. Partialtiles selectedhaveallreflectedaproncoordinatesvalid. Sourcewidth33smallshapeindexlimitation notGPUmeasuredorwholeCLIcertified. One separate sharedannotationPythonB2parserrepair retained; all3devicecandidate/checkfirstpass. Meansseconds {"cuda": 3.2487199999999996e-05, "cuda_jl": 4.2008900000000004e-05, "ka": 5.221745000000001e-05, "jacc": 0.0001011304}.
