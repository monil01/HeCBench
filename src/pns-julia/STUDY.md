# PNS: stochastic Petri-net batch, defined initialization

## Contract and science

PNS simulates random transition firing on a periodic staggered Petri net. Each enabled transition moves tokens from two input places to two output places, with source parity-based conflict priority and barrier-separated phases. Rewards are marking second-moment variance (`E(mark²)-1`, assuming mean1) and maximum marking. No specific chemical/biological application or fitted stochastic model is supplied.

Primary named case: **N1024, S312, four trajectories, original seeds5489–5492**. One complete source256-thread persistent kernel pertrajectory executes grid initialization,624-word untempered MT update, three256/256/112 transition waves, full marking scan and reward reduction. Source stepcounter advances312 after624attempts; this is not312firings. Full originalN22/S100000/T1000000 and catalogueN5000/S100/T100 are **unverified**. Four trajectories cannot establish ensemble confidence, stationary distribution or long-time model validity.

Native `conflict_array` is allocated but read before initialization. The named harness explicitly zeroes it outside ROI before warmup/trials acrossall4models: **native_reference_repaired=true**, excluded from untouched-native aggregates. Original GPU kernel bodies remain unchanged; this host initialization repair does not certify original CLI allocation behavior. Only actual4trajectorycapacity is allocated, ratherthanunused750MBbudgetcapacity, outsideROI.

Independent CPU UInt32 untemperedMT and barrier-phase simulation proves unique individual-byte decrement/increment destinations at all12primarywaves. OriginalN22firstwaves have36–60duplicatewrite destinations on same seeds: a static source race witness, not an observed sanitizer failure. Primary proof does not generalize to other seeds/sizes/long trajectories. Source RNG final local tempering is not storedback; actual untempered MT semantics preserved.

## Numerical and performance evidence

Final `20261018T001703967157Z_pns`, config `study_config.json`. All20 artifacts are byteidentical acrossmodels and independently revalidated in `audit_saved_outputs.py` (NumPy2.5.4). Each has16777228Float32 values: actual4parameterprefix thenall4 marking/conflict arrays andvariance/max rewards. ActualGPUmarks remainInt8bytes,conflictsInt32,MTUInt32,varianceFloat32. Small metadata/state integers are serialized exactly, not emulated as floating arithmetic onGPU. Every2097152markings/trajectory, token total2097152, squared-marking sum, variance andmaximum2 checked exactly; tolerance0.

Five full in-process trials follow one complete kernelwarmup. Mean uses runs2–5, no outlier removal. Timed ROI includes full source-equivalent persistent kernel and global device completion; conflictzero/reset/transfers/oracle/serialization outside.

| Native CUDA | CUDA.jl | KA | JACC |
|---:|---:|---:|---:|
|455.969µs|472.612µs|500.977µs|575.130µs|

Native reuses624UInt32sharedwords(2496B) for reward buffers; existingCUDAjl allocates separateMT/sum/max(4544B), retained byKA. JACC packs the same4544B intoone1136UInt32arena withnonoverlapping offsets/Float32reinterpret andadds templatecopy/barrier. All deviceindices/shared/barriers are ownframeworkAPIs; CUDA hostallocation/completion glue onlyNVIDIA-tested. Fourblocks underutilizeproductionparallelism; memory/cache/occupancy counters unmeasured. No framework-only causal performance claim or wholeensemblethroughput claim.

## Reference correction and attempts

Historical CUDA.jl performs wholeUInt32 RMW perchar update. Concrete primary seed5492/wave11lanes131/220 writebyte793943/793942 insideword198485 (`diagnostics/packed_word_witness.json`). This addscrossbyte race risk absent nativeindividual-byte stores. **ActualhistoricalGPUdiagnostic onprimarycase passed**, so no fabricated observed numerical failure. Reference proactively changes onlyplace load/storehelpers toInt8 accesses; originalpersistentkernelbody is byteidentical (`reference_repair_provenance.json`).

Reference onecandidate/oneactualdevice-JITcheck. KA two programcandidates/processchecks: initial expandedhelperreturns cause frontendmacrorejection **beforedeviceJIT**; repairedcandidatepassesoneactualdevice-JITcheck. JACC firstcandidate/checkpasses. Onegenericpersistentkerneldefinition each; internalcompilerpasses/specializationsunknown. Old KA source/log retained, no tolerance/algorithm repair or fastest-run selection. Priorreference/KAknowledge used; notcontext-free firstprogramsuccess.

Prospective usage windows include context/cache/reasoning/coordination and overlapthebroadworkflow, notadditiveexclusivebilledcost. PNSreference/broadwindow includesbriefparticlesunusedwrapperdigestrepaircoordination, disclosedinterleaving. Sourcechars/4proxies separatelypartial. Alltraits/modelreviews/failureevents/status/attempts/usage/checkpoint/paperobservation ready; rootcoordinates separatevariantcommits. Nextqueuepso.
