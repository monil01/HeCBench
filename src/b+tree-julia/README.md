# B+ tree point and range queries

Bothsourcekernels arefullyported, cooperative256threads/queryblock, on17/4096/65536recordtrees withheights0/1/2 and33/257/1024queries/kind. SourceCPUinsert/transform_to_cuda createsgenuineB+trees from8191permutation/spacedintegerkeys andnonidentityrecordvalues. Originalmil/commandassets areabsent. Allconsumedsourcefields retained; serializerdefinesdeadpadding/unusedindices aszero andunusedkeysINTMAX. CUDAphysicalknode2068bytes matchesJuliaflat517Int32words exactly.

## Explicit conditional native reference

OriginalCUDAfiles unchanged under originaltree andarchived `reference_history/`. Original12.0/12.9 queryprogramsfail fullindependentchecks aboveheight0. PTXshowsrestrict-sensitivecachedcurrent-node/keyreads acrosscross-lanemutablepathupdates. Minimalfactorcontrol: nonrestrict0wronglanes; restrict255/256wrong; volatile+restrict0. The correctedowned `study_repaired_kernel*.cuh` removesonly7restrictqualifiers fromsharedmutablepaths/rangeRecstart, notreadonlytree/record/querybuffers. No volatileintrinsic/math/barrier/layoutchanges. Hashes/witnesses in native_pointer_contract_witness.json andreference_provenance. This isconditionalcorrected-native coverage, excludeduntouchednativeaggregate, notblanketCUDAcompiler/memorymodelfailureclaim.

## Full correctness/timing

Fivecheckedtrials save382846Int64values/trial: actualallGPUknode/record/query/initialpath/outputreadbacks (365644wordexactprefix), everypointanswer/rangeindex+length/mutablepathstate, andguards forall14buffers. Readonlyinputs recheckedpostkernel. Independentordered-mapvalues/missingkeyretention/inclusivesortedrankranges plus completegraphreachability/partition/leafrecordcoverage andpathoracle. Allintegeratol0rtol0; all20acceptedcaptures bitidentical. FinitekeysstrictinsideINTMIN/MAX; rangeendpointsalwayspresent, missingpointanswersretaininitialpoison-777.

Eachtrialtimes6sourcepoint/rangekernel/contextcompletionROIs, CPUbuild/reset/inputcapture/checkexcluded. Oneeachkernel/casewarmupoutsidealltrials, mean2–5. Oneblock/model/shortkernels; no originalfullmilCLI/productionthroughput orcrossbackendproof. DevicecomputeInt32/Int64, Bool/paddingnodefieldsunusedbykernel; nofloatingpointrooflineclaim.

## Generate the required fixture

Run fromrepositoryroot withnvcc12.0/g++available; generatorinvokesCPUtreefunctions only (CUDAwrapperobjectslinkedbutnotexecuted):

```sh
nvcc -O3 -std=c++17 -arch=sm_70 src/b+tree-julia/study_fixture.cu src/b+tree-cuda/kernel/kernel_wrapper.cu src/b+tree-cuda/kernel/kernel2_wrapper.cu src/b+tree-cuda/util/num/num.c -o /tmp/btree_study_fixture
/tmp/btree_study_fixture src/b+tree-julia/fixtures.bin
```

asset_provenance.json records exactfixturebytes/hash/generator. Missing/short/wrongshapefixtures fail; nosyntheticdefaultfallback. Compilewarnings come fromunusedlegacyCLIformat/scanfcode, preservedlog, notgeneratorfailure.

## Replay

PinnedJulia1.12.6,CUDA.jl6.2.1,KA0.9.45,JACC1.4.0,nvcc12.0/sm70. Requiresgeneratedfixtures.bin orcommittedexactasset; recordedabsolutepaths/projectenvs mustberelocatedelsewhere. Nativebaseline isexplicitlycorrected:

```sh
python3 tools/verify_portable_julia.py --config src/b+tree-julia/study_config_native_repaired.json
python3 tools/verify_portable_julia.py --config src/b+tree-julia/study_config_refs.json
python3 tools/verify_portable_julia.py --config src/b+tree-julia/study_config_ka.json
python3 tools/verify_portable_julia.py --config src/b+tree-julia/study_config_jacc.json
```

Freshreplay removesreusefields orpointsnextconfigtofreshpriorresult; localhistoricalrawcaptures arenotcheckoutassets. Centralverifier ownsGPU6lock. Oldoriginal-native/controlconfigs intentionallydemonstratefailed/unacceptedreferencebehavior andmustnotbepromoted. `study_audit_original_failure.py FAILED_TRIAL_BIN` performsCPU-only fullartifactdiagnosis. Per-modelproof/accounting, canonicallocaltraits/status/effort andpaperobservations provideactualsource/attempt/tokenunits; cached/reasoningsubsetsnotadditionalcosts.
