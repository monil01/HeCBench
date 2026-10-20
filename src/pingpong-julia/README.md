# Pingpong — verified named two-GPU MPI/NCCL study

Both immutable originalCUDA +1 kerneldefinitions retained. All3targetversions ownframeworkallocation, two translatedentry kernels, and genuine ownJulia nestedMPI/NCCLloops. SharedCABI exposes individualcommunication operations+checking, nohiddenCroundtriploop orGPUkernel. Class mixed_kernel_communication_pipeline, with framework_device_kernel components and vendor_library_interop communication. **Timedauthoredkernelcalls0**: source5incrementvalidationroundtrips areoutside50roundtripcommunicationROI.

Named65536/262144/1048576/257Float64 sizes, bothmethods, twolocalranks physicalGPUs5+6 only. Originalup-to1GiB sweep unmeasured. Exactdyadicperrankfixtures differoriginalzeros; all actualGPUinputs, postincrement/postcommunication buffers and all64guardwords retained. All20checkedtrials passindependentNumPyoracle and exactdualGPUreferences. Onefullwarmupblock outsidefivechecks, means2–5. Metric is **sum8rank0elapsed/(2*50) per-case means**, notisolated1Mbandwidth. Individualbothrank/casetimes retained.

GPU5idlepreflight/UUID/order and liveexternalPIDmonitor, plusbefore/aftercases/trials checks, allrecorded. Guardkills onlyownverifiedjob ifexternalworkappears. JACCspecialrankbootstrap records exactPIDbeforeframeworkstartup, execpreservesPID. AllacceptedseriesexternalPIDlists empty. FailedJACCstartup-registration/API/stream errors and boundedfirstNCCLwait source/logs preserved. Correctselecteddevicestream+eager2kernelsbeforecommunicationsetup yieldsall5; mechanismofearlierhang notcausallyisolated. Noframeworkunsupported/performancepenaltyclaim. Noexternalprocesssignals, sudo orOSmutations.

## Reproduce

Prerequisites: pinnedJuliaenvs; nvccCUDA12.0, availableHPCXOpenMPI4.1.7rc1/CUDA-awareUCX andNCCL2.26.5. ActualDSOpaths/hashes and perprocesslibraryversions retained. ConfigsuseinstalledHPCSDKpaths; adaptandrecordnewenvironment ifrelocating. NeedtwoauthorizedidleGPUs; recordedcase5+6mustnotbechangedwithoutscopeauthorization. ExistingcentralverifierownsGPUlock, noouterlock.

`python3 src/pingpong-julia/run_study.py --config src/pingpong-julia/study_config_jacc.json` (correspondingnative/CUDAjl/KAconfigsavailable). Configsbindallsource/runtime dependencies. SharedABIbuildbyexistingnvcc isunprivileged, no systeminstall. WrapperusesexistingSTATE=/tmp/hecbench-maps-537-state andcanonicalrunaliases/storageprovenance, nofrozencapturedeletion/move. Preserve/tmp evidencebeforecleanup/reboot. All20primarycaptures~2.64GB, noexternalinputfiles.

CPU audit: `PYTHONPATH=/tmp/stencil_ports_numpy_226 python3 src/pingpong-julia/audit_captures.py --model jacc --result RESULT --receipt RECEIPT` (NumPyrequired; pathislocalinstalleduserdependency). Fullsource/artifact auditreceiptandpermodelcaptureauditsretain actualbytes, nothash-onlyclaimedproofs. Do notcommitcompiled.so/executables/rawcaptures. Essentialsourcefixtures are deterministic formulae in study_common*.jl and study_comm.cpp, noadditionalbinaryassets.

Timingincludesper-operationhostcommwrapper/loopoverhead, notonlyDMA. Excludeduntouched-native andsingleGPUaggregates. Reviewedhardwarecommunicationfamily, no scientificsimulation, inter-nodescaling, crossbackendportability ormeasuredroofline claim. Actualcontextinclusive usage windows includecoordinationandsharedintegrationrepairs; nestedverificationoverlaps, noexclusivecoding/billingclaim.
