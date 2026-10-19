# K-means source-faithful scoped pipeline

Both original CUDA device kernels are included unchanged by `study_native.cu`. Original files are byte-preserved in `reference_history/`. Three genuine Julia versions implement transpose and nearest-center kernels and the complete host iterative centroid/RMSE/restart/k-sweep pipeline. No CPUassignment or vendorclustering substitutes for GPUassignment.

## Named inputs and source scope

The original KDDdataset is absent (only DVCpointer exists). Deterministic Float32dyadic fixtures are generated directly by `study_support.h` / `study_common.jl`:17points×3features/k2–4/2restarts;257×4 and4099×7 bothk5–15/10restarts (the sourceMakefile algorithmic sweep). The sourceinitial[] swap/decreasingcounter persists acrossrestarts, resetsperk; n>=k*restarts avoids originalunderflow forsmallcontrols. All actualconvergenceiterations retain changedmembershipthreshold.001 andsource `loop++<500` stoppingpolicy (atmost501iterations), strictfirstindexties, emptycenterretention, sourceFloat32sums and optionalCPU RMSEselection.

The source assigns Float32 RMSE to `int rmse` before bestmodelselection. `study_selection_witness.cpp` invokes original `rms_err` and demonstrates1.41421354 versus1.23743689 bothtruncate1, causing the selector to miss lowerrealRMSE. This is a selectorcomponent witness, notclaimthese artificialstaticcenters areLloydfixedpoints or allCLI answerswrong. Nativebehavior ispreserved, known-defectscientificselection stratum excluded fromfaithfulscientificoperatoraggregates; no correctednativeclaim. Finalreturnedcenters sourcepolicy islastk/lastrestart, notnecessarilybestselectedmodel.

## Complete evidence

Each offivecheckedtrials saves1,881,560Float64encoded values: exactactualdeviceinputprefix29,788words, alltransposedfeatures/periterationdeviceclusterreadbacks/everymembership/updatedcentroid/delta/iteration/restart/RMSE/selection state, and allbuffer guards. ActualFloat32/Int32fields arelosslessly encoded; physicalinputbits checked separately insideharness. IndependentFloat64nearest-center/firsttie checksallassignments, independentFloat64means checkallupdatedcentroids (max4.4505e-7, predeclared2e-6). Primarydual atolandrtol1e-6, exactprefix/guards/IDs; actualcapturedall4outputs arebitidentical.

## Measurement

Devicebuffers preallocated; hosttemporaryallocations insidecentroidupdate remain timed. ROI sums source transpose plus eachclusterupload/GPUassignment/memberdownload/hostupdate andCPU RMSE/selection; initialcenterselection, tracecapture, deviceallocation and inputinitialization outside.717actualkernellaunches/trial. Bothdevicekernels andtimedCPUhelpers warmedoutsidealltrials; retainmean2–5. OneearliernumericpassCUDAjlseries hadmissingCPUhelperwarmup; rawoutputs/logs retained andexcludedfromprimarytiming, correctedfreshseriesused. CUDAjl acceptedtrial5hostruntime spike retained, notremoved; causeunprofiled.

This preallocated stage-sum isexplicit timing/lifetime intervention, notsourceallocation-inclusiveCLI throughput. It includes CPUloops/runtimeallocations/GC, notpuredeviceabstractionlowering. Onefive-trialblock/model andmissingdataset limit application/performanceclaims.

## Replay and witness

No externalgeneratedfixtureasset needed. Requires nvcc12.0,g++,Julia1.12.6,pinnedCUDA.jl6.2.1/KA0.9.45/JACC1.4.0; recorded sm70GPU6. Recorded absolutepaths inconfigs must be relocated. Centralverifier providesGPUlock.

```sh
g++ -O3 -std=c++17 src/kmeans-julia/study_selection_witness.cpp src/kmeans-cuda/rmse.cpp -o /tmp/kmeans_selection_witness
/tmp/kmeans_selection_witness
python3 tools/verify_portable_julia.py --config src/kmeans-julia/study_config_native.json
python3 tools/verify_portable_julia.py --config src/kmeans-julia/study_config_refs.json
python3 tools/verify_portable_julia.py --config src/kmeans-julia/study_config_ka.json
python3 tools/verify_portable_julia.py --config src/kmeans-julia/study_config_jacc.json
```

Fresh replay removes reusefields or updatesnextconfig toitsfreshpriorresult; historical localrawcaptures are not checkoutassets. Paperobservations and pervariantaccounting/verification provide observedusage windows, sourcehashes, exactattemptunits and scientificlimits.
