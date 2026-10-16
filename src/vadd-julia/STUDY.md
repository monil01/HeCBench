# VADD matched configuration

Named aligned256x256 half precision fixture,100internalrepeats; other dimensions/tails unmeasured.

Native packedhalf2/vectorized TVmapping differs inheritedJulia scalar/vectorizedalias and32values/thread TVmapping.

Original CUDA.jl inputsign convention differed; sharedfixture now matches libc srand42 native. Original devicekernels preserved.

All three native half addition cases run unchanged, originalJulia cases retained and ported. Full196608 half outputs losslessly widened only for binary artifact comparison outside timing. All five trials exact against native and CUDA.jl, atol1e-2/rtol1e-5, mean of lastfour. JACC initial automatic sharedmemory policy is baseline, not a required kernel resource.
