# BMF: complete UInt32 Boolean factorization optimizer

See SCOPE.md for three named17×33,65×97,257×529fixtures, latentdimensions5/13/20,7/105/21iterations,2sequentialrestartssingleactive slot. Includesoriginalinitialization, alternatingrow/columnproposals, Metropolisrule, distancepolling/cooling/stopping, bestfactorselection/copies. MissingMNIST/default2slotconcurrentCLI/10000iterationsunmeasured. Nooptimality/realdataqualitycertificate. Originalnative treeimmutable andarchived original_native/ withGPLlicense.

## Required assets and clean preparation

Onlyonebinaryinputasset: fixtures.bin36588bytes, hash inasset_provenance.json. IndependentPythongenerator hasnodependencies beyondstandardlibrary andwritesfixtures+manifest:

```sh
python3 src/bmf-julia/generate_fixture.py
g++ -O3 -std=c++17 -shared -fPIC '-DBMF_ROOT="/home/imo/HeCBench_julia/src/bmf-julia"' src/bmf-julia/study_check.cpp -o src/bmf-julia/study_check.so
```

ReplaceDBMF_ROOTwithactualcheckoutabsolutepath onrelocation. Generatedstudy_check.so islocalruntimechecker, notsourceasset; rebuildbeforeJuliaexecution. Configsbinditsactualbytes, allotherdeclareddeps, originalheaders, andfixtures. Missing/wrong-sizedassetsfailratherthanfallback. Allrawtrialcaptures stayin.porting-state, notcheckoutassets.

## Verification and timing

Complete7544Float64-encodedvalues/trial (losslessUInt32/Float32expansion),5344exactactualGPUinputprefix. Fullactive/bestfactors/allrestartdistance/guards,readonlymatrix, scalarindependentPythonreplaychecks everycandidate. All20acceptedcaptures identical. Float32positiveMetropolisdecisionmargin inmanifest, no universaltranscendentalerrorbound.

Fivecheckedtrials, keepmean2–5. Fullsourcepipelinewarmupoutsidealltrials. ROIincludesfactorinitialization+hostthresholdformula+per-threaddevicelog2, allupdates/distance, hostsync/printing/cooling/polling/bestD2Hcopies. Allocation/matrixupload/capture/checkoutside. SourcehostOpenMPsingleworker versusJuliaserialhostcontrolisexplicitmappingdifference. Eachtrial556authoredkernellaunches/fourentrydefinitions/model; service/runtimecalls arenotauthoredkernels. KA/JACCgenuineownframeworkdispatchbutCUDAshared/barrier/shuffle/atomicAPIsretained: hybrid_device_intrinsics, no crossbackendclaim. CUDAjlusesownexportedCUDAAPIs/framework_device_kernel.

Freshreplayfromroot:

```sh
python3 tools/verify_portable_julia.py --config src/bmf-julia/study_config_native.json
python3 tools/verify_portable_julia.py --config src/bmf-julia/study_config_cuda_jl.json
python3 tools/verify_portable_julia.py --config src/bmf-julia/study_config_ka.json
python3 tools/verify_portable_julia.py --config src/bmf-julia/study_config_jacc.json
```

CentralrunnerownsGPU6lock; noouterlock. Configshavehistoricalreusepaths; freshcheckoutmustremove reusefields orpointsubsequentconfigtofreshpriorresult, andrelocateabsolutepaths/Juliaenvs. Julia1.12.6/CUDA6.2.1/KA0.9.45/JACC1.4.0,nvcc12.0/sm70.

## Preserved work-alignment feedback

InitialJuliahostthreshold/initializationdepthworkwasomitted/hoisteddespiteexactfactoroutputs. OldCUDAjl/KArawseries arenumericallyvalidbuttiming-invalid; hash-boundstudy_adjudications.json excludesaggregates, reference_history/ preservescode. JACCinitialqueuedraftcancelledbeforedeviceprocess. AllthreecorrectedJuliafive-trialseriesnowauthoritative; originalnativeunchangedreused. JACCModule-not-callable hostconstructorfailure preserved; explicitinit_backend/launch_specfactoryfixpassesbeforedevicebodychanges. NoGPUlanguageunsupportedclaim.

MeasuredKA/JACCtrial5spikes retained. Onefive-trialblock/model, nooutlierdeletion/stableframeworkranking; hostgenericdispatchhypothesis unprofiled. Tokenactivitywindowsareactualcontext-inclusive/disjointintervals+separatesharedrepair, notexclusivecoding/billing/historicalreconstruction, excludedordinarycostcalibration. CanonicalRQ1–5observations/traits/effort/proofs recordactualunits.

## Separate diagnostic control

`host_specialization_predeclaration.json` and `host_specialization_results.md` preserve a one-line host function parameter specialization experiment. `study_jacc_specialized.jl` is outside the primary child folder, so primary source hashes remain unchanged. `study_config_jacc_host_specialization.json` reuses the exact original reference captures and runs a fresh five-trial JACC block. Its role is `diagnostic_tuning`; it must not replace primary data or enter the primary cohort. Read the control's limitations before interpreting its lower mean. Tuning tokens are recorded separately.
