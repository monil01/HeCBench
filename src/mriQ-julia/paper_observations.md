# MRI-Q prospective observations

All four versions pass fullactualinput/Phi/complexQ/guard/Float64oracle checks, five trials each. CUDAjl/KA/JACC artifacts are bitidentical. Native/Juliadifference max3.814697265625e-5; nativephysicaloracle max1.92697859788e-4, Julia1.85068465257e-4 under predeclaredatol1e-4+rtol2e-5. No thresholdchanged.

Meanlastfour ms: native0.3019005,CUDAjl0.8470515,KA0.99875,JACC0.979135 for11explicitfragments, notoriginalMRIdefaultdata/wholepipeline. Nativefastmath/constantmemory vs Julia math/globaltile recorded as covariates, withoutcausal/rooflineclaim.

JACC candidate1 rejects host-array arguments duringGPUCompiler validation; candidate2 switches onlyallocator toJACC.array. CUDAjl/KA authoredonecandidate each; actualprocesschecks3/2 duecoordinatorreusekeymistake, distinctfromrepairs. Prospective phasecounters andclocklag retained; rootinterleaving excludesclean costfit. Allcontrols,sourcehashes,exactscopeandRQlinks inpaper_observations.json.
