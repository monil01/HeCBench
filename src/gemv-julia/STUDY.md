# Quantized GEMV study

## Actual source contract

The active native main invokes signedINT8 and packedunsignedINT4 matrix-vector products. The separate rawFP16 GEMV call is commented out and remains unmeasured. Both active modes are covered at source-defaultN512/block32x4. Low/high INT4 nibbles represent consecutive columns; no expanded4-bit surrogate storage is used. Vector,scale,zero_point and output areFP16, with FP32scalarFMA accumulation/reduction. Scale is.0625, actual stored halfzero_point is.01000213623046875. This is scalar dequantized lowbitlinearalgebra, not integerdot/tensorhardware or FP16accumulation. Source identifies no trainedmodel/scientificdataset.

The historical Julia port used FP32vectors/parameters/output and different stridedlane ownership. That source is archived. Current genuine CUDA.jl reference restores the nativehalfcontract and consecutive8-term INT8/16-term INT4 chunks perlane, postscale,32-lane treeorder and finalhalfrounding. The widened old implementation is a different precision/mapping scope, not declared mathematically wrong within its own scope.

## Numerical evidence

Every trial checks all512 nativeFP16outputs, losslessly widenedFloat32 only outside timing. An exact CPUemulator reproduces FP32FMA laneorder,scale,32-lane binarytree and finalRNEhalfconversion. Native,CUDA.jl,KA,JACC allmatch exactly. A separate independentFloat64 directdequantizeddot check retains originalN512 absolute.125 criterion; maximumerrors are.01544448 INT8 and.05975083 INT4. This intended nativehalfrounding is not a translationfailure. All outputs must be finite. No arbitrarydimension/vectorchunk validity is claimed.

## Mapping and compiled resources

All4use128groups32x4,4rows/group. Native loads packed8/16-value chunks through vector structs and reduces with warp shuffles. Julia-family scalarloadsthose same orderedchunks anduses a32x4Float32sharedtree withbarriers. Thus arithmetic is matched but memory/reductionprimitives differ.

Actual cudaFuncGetAttributes in eachnative studyprocess reports8192B staticshared and32registers even though the selected32-lane branch returns before accessing the declared64x32 warp-sumsharedarray. CUDA.jl/KAuse512B staticshared/local; JACC explicitlyallocates512B andstages a32x4zero template, adding a barrier. Occupancy/resource effects are unprofiled; no slowdown cause is inferred. Native unusedsharedreservation, vectorloads/shuffles versusportabletree are explicit confounds.

## Timing and interpretation

Ten kerneloperations pertrial, explicitJITwarmup,5synchronizedinprocesstrials, meantrials2–5. Resets, transfers, half-to-float artifactconversion andbothCPUoracles are excluded. INT8matrix256KiB/INT4matrix128KiB fitsV100L2; other sizes/blockshapes and scientificinputregimes unmeasured. MixedFP32FMA plusintegerunpacking/halfconversion requires an appropriate performance model; FP16tensorpeak or a generic4-bit arithmetic ceiling would misdescribe actualcompute.

Each target passed firstprogramcandidate in2actualmodechecks,0repairs. Current reference is1separatelycostedsemanticprecision/mappingrepair,1candidate/2checks/2nativebuilds. Historicalgenerationcostunknown. JACC intentionallyreuses successfulKAmapping. Prospectiveconsumedusage includescache/context/coordination, partialsavedsource proxies are separate, broadworkflow overlapschildphases andisnotadditive ormonetarybilling.
