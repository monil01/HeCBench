# nonzero verified configuration

nonzero_i32_256x256_stable_coords_count_pipeline_prealloc_v1

NativeCUB vendorprimitives vsoriginalCUDA.count/findall vsnewframeworkcustomhierarchicalprefix/scatter, notintrinsic-matchedalgorithm.

Customper-elementblockprefix O(256N) andblockoffset O((N/256)^2), correctscalableatnamedsize butnotoptimizedwork-efficientscan.

NativeCUBworkspacepreallocated outsideclock; CUDA.jl internalfindall allocation remains insideclock. Namedregiondoesnotmeasureoriginaltempallocationoverhead.

Coordinateorderpreserved exactly, never sortedaftertiming tohideunstablecompaction.

Fullcount+D2Hcount+stablecoordinatepipeline; input/outputallocationsoutsideclock. NativeCUBtemp preallocated; historicalCUDA.findall still internallyallocates, explicitAPIpolicyconfound.

Five in-process trials, all outputs checked against both references and the configured CPU oracle. First trial time excluded; arithmetic mean of last four. Raw samples/configuration/provenance remain linked in per-variant verification.json. Partial token estimates are not total usage.
