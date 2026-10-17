# Group normalization study

## Scientific operation and coverage

All three native modes are measured separately: scalar forward, four-component forward2, and backward input/channel gradients. The named Float32 NCHW case is B4,C32,H16,W16,G8: four consecutive channels and1024 pixels pergroup,32groups. Group population moments use E[x²]-E[x]² andepsilon1e-5, with per-channel scale/bias. The source references llm.c/image-channel normalization but supplies no model/scientific dataset; no application-accuracy claim is inferred. Source B32/defaultlargeimage and highoffset/lowvariance regimes remain unmeasured.

Forward checks all32768outputs+32means+32inverse-standard-deviations. Backward checks32768inputgradients+32weightgradients+32biasgradients:32832values in eachmode, everytrial. IndependentFloat64 formulas and8 centered input finite differences validate mathematical gradients. All are finite and satisfy strict absolute1e-3, stronger than sourceout/dweight/dbias1e-2 anddx1.0. Maximumforwarderrors are below2.4e-7; backward below2e-6. Savedbackward inputs use the same canonicalFloat64 moments roundedFloat32 in allmodels; this isolates backward rather than claiming an end-to-end training pipeline.

## Mapping and reference correction

The original Julia forward had genuine math but seriallane0stats; its forward2 label aliased the same kernel. Archived original source is retained. The current reference restores a distinct four-component path and parallel paired full-block trees. A potential scratch-read/overwrite hazard in the old reducer was found bysourceinspection, not observed as a runtimefailure; an explicit post-readbarrier removesit before reuse. This is one current source revision, not fabricated repeated compiler repairs.

Native uses cooperative-group warp reductions andfloat4 loads. Julia-family uses manual1024-lane paired trees and scalar component loads; these are agent-selected mappings, not framework-optimality claims. Native source declares256B sharedpartials; Julia-family allocates8192B. Native actual compiled resource attributes/registers/occupancy were not measured here. Backward source uses warp-per-channel accumulation in this small-image branch; the larger-image alternative remains unmeasured.

JACC explicitly allocates8192B and stages2048zero-vector values, adding an initialbarrier. Pinned JACC1.4 has an unimplemented shared-matrix copy branch for one-dimensional threadblocks. A supported flatvector backing represents the two logicalcolumns at offsets0/1024; this nonblocking API boundary is documented with backendsourcehash and caused no failedcall. Native/portable mapping, scratch and barrier differences confound attributing gaps solely to framework overhead.

## Timing and effort

Each selectedstage runs in its own process with explicit JITwarmup,5 synchronized inprocesstrials, meanlast4. Forward/forward2 eachrepeat10 overwrite operations. Backward repeats1 complete gradientoperation; resets prevent accumulatedatomicgradients from changing the mathematical result. Allocations/transfers/canonicalmoments/Float64math/finite differences/artifacts are outside timing. The boundedsynthetic32768-value fixture fitsV100L2, so no largetensor/DRAM bottleneck claim is made.

Each KA/JACC program passed candidate1 with3 actual modechecks,3kerneldefinitions and0repairs. Currentreference is separately1revision/3checks/3nativebuilds. Historicalgenerationeffort remainsunknown. JACC intentionallyreuses KAfeedback. Prospective consumedusage includescache/context/coordination; partialsource proxies stay separate, broadworkflow overlapschildphases andisnotadditive ormonetarybilling.

The selected case has only 32 independent blocks. Larger batch/group-count regimes and full-GPU scaling remain unmeasured.
