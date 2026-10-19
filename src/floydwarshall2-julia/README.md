# ECL-APSP: full blocked64 all-pairs shortest paths

This port retains all five source GPUentrypoints, 64×64 tiles, 1,024threads, warp shuffles, sharedmemory barriers, staticallyexpanded64loops, and the complete source driver. Seven named weightedCSRgraphs cover33/64/65/129/257/513/1025nodes. OriginalCollegeMsg dataset/default1000repeats are not certified.

`contract.json` predates the first native measurement. Independent arbitrary-integer Dijkstra checks every distance, rowworkspace and tile-local-transposed columnworkspace exactly. ActualCSR/scalars/initialmatrices/readonly/guards are read back and checked. All20saved fullarrays have the same SHA256. Initialauxiliarymatrices are definedINF scaffolding outsideROI; no sourceGPUbody ormathematicalalgorithm repair.

The source initializesR3 thencomputesR3 on one matrix. Latercomputes operate on an alreadyclosed matrix; do not interpret the source mean as threecoldfirstsolves. The measured interval sums seven source init/compute regionmeans, with context completion. Per-case split times remain in rawlogs. Onefullwarmup precedes5checkedtrials; discardonlythefirsttime/meanlastfour. FileI/O/allocation/transfers/observation/oracle are outsideROI.

CUDAjl owns source-entrypoint kernels. KA uses KA allocation/dispatch and JACC uses JACC arrays/explicitlaunchspecs; both call translated CUDAwarp/barrier/staticshared helpers and are hybriddeviceintrinsics implementations. There is no foreignnativeGPUdelegation. Native2×16KBsymbols map to distinctranges in one32KB CUDAsharedarray. Source compiler launchbounds/CUDAjlregistercaps and normalKA/JACC compilerpolicies remain performancecovariates, without profiling orcrossvendorclaims.

Run `python3 generate_assets.py` to reproduce the6.8MB independent fixture exactly; it remains local with its SHA/recipe. Use `prepare_cold_config.py` to prepare a fresh4modelconfig in another checkout, then run the studyverifier. Project/toolchain paths are machine-specific and must be installed or adjusted. Preserve the staticcontract, fixtures and all executed source content.

The first submitted CUDAjl helper had a hostparsererror pairing a separate-lineelse; its entire submitted source and diagnostics are archived. Candidate/program/JIT/generator counts are separate. Actualtranslationusage is observedcontext-inclusive input+output, with cachedinput/reasoning subsets and inheritedcontext/orderreuse. Sharedphases overlap and cannotbeadded; historicaltokens are not recovered.
