# Braycurtis: distance and gradients

RQ1: All20 full162154-value captures meet the declared contract, including95346 exact actualdevice-inputprefixvalues, allfive survivingoutputarrays andguards. CUDA.jl/KA/JACC are bitidentical; native differs by at most5.96e-8 dualabsolute/0.0003617 relative, interpreted under combinedabs+rel tolerance. Independentactual-inputNumPygradient math passes everyfield; all56positive and1848signed-case scalarpartials pass centralfinite differences. Native4devicebodies are exactsourceextractions throughreviewedrowmajoraccessorshim: originalPyTorch/autogradintegration is unavailable/unverified, excludedfromuntouchednativeaggregates, not a mathematicalnativecorrection. F16dispatch/largeoriginalshapes remainunmeasured.

RQ2: Eachtargetfirstcandidate/JITprocess passes;4algorithmdefinitions+1untimedobserver,2precisions×3shapes are notattempts. CUDAjlcreatesdevice/hostsharedhelpers; KAthenJACCinherit. Prospectiveobservedstreamcounters retaincached/contextreuse andsamebenchmarkcoordination. Oneattachmentmetadatakeyerror issharedtoolingfailure, not devicefailure.

RQ3: Mean2–5 composite seconds CUDA0.0003419135, CUDAjl0.0004425645, KA0.000477117225, JACC0.000566159025. Sum6 cases(forwardR10mean+3kernelbackwardR10mean), withwholedevicecompletionbetweenforward/backward; setup/copies/checks excluded. No perprecision/single-size performance inference or causalabstractioneffect.

RQ4: AllthreeGPUimplementations areframework_device_kernel: scalarabs/division/sign,globalrowmajorarrays,256threads/group, no sharedmemory/atomics orforeignCUDAdeviceintrinsics. Sourcebyvalueaccessor vsflattenedJuliaview/launchwrapper andstreamboundaries remaincovariates. No profiler/rooflinecoordinates.

RQ5: Thisfull4kernelarithmetic contract is expressible without backendextensions across thethreeJuliafrontends, but no crossvendor run was made. Selection must retainnativeintegrationqualification, unmeasuredF16/dataset/singularpolicy andcompositehostwrapperboundaries.

Additional CPU-only saved-series audit in the annotation window checked all25 ordered trial pairs under the declared per-precision field budgets, including the stricter F64 criterion and exact input-prefix bytes. All pass; no GPU execution or original runner result changed.
