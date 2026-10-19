# MPI all-reduce observations

RQ1: The named context-fenced protocol passes all four versions, fivecheckedtrials each:5,269,603Float32values/trial, includingexact2,106,001wordactualdeviceinputprefix, fullboth-rank sums/receivedchunks/guards. Only2ranks onphysical5+6 aretested. Both KA/JACC own their gridstride additionkernel; shared CABI wrapsMPIcommunication only.

RQ2: Oneauthoredmathematicalkernel perversion. CUDAjl needed5jointMPIchecks (10rankprocesses) plus2no-authored-kernel diagnosticMPIjobs; infrastructure/frontend/signal/framing/completionfeedback areseparate fromGPUmathcandidatecount. JACCneeded2hostlaunchwrappers after a globalLaunchSpeccapturedwrongdevice-stream. DisjointactualCUDAjltranslationwindows sum6,422,310context-inclusivetokens; nativeadaptation isseparate. Completeworkflowtokencoverage isnotclaimed for a same-roundsharedphaseBEGIN; nofabricatedcorrection. Calibrationrequires>=8samples/stratum, theseareobservations.

RQ3: Means2–5: native0.0035921135s, CUDAjl0.004104573s, KA0.004497446s, JACC0.0049733095s. Theseare sum6case max2rankdurations of3preallocated ringcalls/case,18additionkernels/rank. Deviceallocation andbarrier/max-clockreduction/captureoutside; MPI/ownGPUadd/contextfencesinside. Separate2GPU/modifiedROIstratum, excluded fromsingleGPU/untouchednativecomparisons. No isolatedbandwidth/causalabstractionclaim.

RQ4: PinnedCUDA.synchronize() waits currenttaskstream, whereas device_synchronize waitscontext. An earlier1Mfull-ringfailure remainspreserved, despiteexactinputs/guards andsmallcasespassing. StandaloneGPU Sendrecv/Irecv-Send-Wait bothpass20times/rank. Strongercontext-fencedv2 all4passes supportstestedprotocol, notproofthatfencesalone causallyexplainfailure. JuliaMPIneededUCX GCsignalhandling/classicallocation settings documented by [MPI.jlknownissues](https://juliaparallel.org/MPI.jl/stable/knownissues/). No genericMPI/frameworkunsupportedclaim.

RQ5: A portable additionbody stillrequires vendor-awareMPI/allocator/device-stream/completionintegration for thispipeline. No crossvendor/multinode/scaling/modelqualitycertificate; source0/8M/64M/536Mmaxsizes andoriginalhugeiterationsunmeasured.
