# K-means observations

RQ1: All four versions agree bitwise on1,881,560recordedvalues/trial, including all transposefields, actualper-iterationdevicecenterinputs, everymembership/centroid/delta, restart/RMSEselection state and guards. The full source k5–15/10restart pipeline runs for257/4099points;17pointcontrol usesk2–4/2restarts to respectsourceinitializationstorage. Float64nearest-firsttie/centroidmeans checkeveryactualiteration; maxcentroiderror4.4505e-7<2e-6. SourceintegerRMSEcast demonstrably losesstrictordering, so nativebehaviorfidelity isdistinctfromscientificmodelselectioncorrectness.

RQ2: Eachmodelhasoneauthoreddevice-sourcebundle/twokernels. CUDAjl has2fullprocesschecks because a sharedmeasurementaudit found CPUhelperJIT warmup missing, corrected beforefreshmeasurement. Oldnumericpassseries remains preserved/timingexcluded. Context-inclusive prospective targetusage andsource/sharedharnessreuse areinJSON; notbilled/exclusivecodingcost.

RQ3: Means2–5seconds native0.0512876305, CUDAjl0.1176698635, KA0.0694700435, JACC0.06678827625. CUDAjltrial5=0.292901259s isretained; itscausewasnotprofiled. AllROIsincludehostcentroid/RMSEwork, copies, hosttemporaryallocation and717kernelcalls; devicebufferallocation/tracecaptureexcluded. No originalallocation-inclusiveCLI timing or pureGPUabstraction/causalrankingclaim.

RQ4: Statisticalclustering, pointwisenearestcenter loops plus hostirregularfeedback; Float32math/Int32members, no device sharedmemory/atomics. No measuredrooflineposition/bottleneck.

RQ5: These named syntheticinputs support operator/pipelinecomparison, notglobaloptimality, KDDlabelquality or bestrealRMSEselection. Originaldatasetabsent; source lastreturnedcentersmaydifferfrombestselectedk/restart.
