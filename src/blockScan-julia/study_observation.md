# blockScan: Compute every exclusiveInt32prefixsum and total of a1024-elementtile

rq_ids: RQ1, RQ2, RQ3, RQ4, RQ5
experiment_ids: E1, E2, E3, E4, E5
configuration_id: blockScan_exact54paths_grid1_threefixtures_v1

## Scientific intent and fidelity

Native CUBexample genuinely benchmarks3algorithms. HistoricalCUDAjl genuinely computesexactprefixes seriallyinthread0, despiteprintingthreeCUBlabels. Portableversions implementonecooperativeBlellochscan and arecomparedwithall3native strategies without claimingidenticalalgorithms.

Scientific task: output[i]=sum(input[1:i-1]); aggregate=sum(input). Requirements and precision sensitivity: Exactintegeragreement forall1025payloadvalues/path and16guardvalues/path; everypartialsumfitsInt32; Int64CPUoracle/differenceidentity.. Community use case: Prefixsum isadomain-neutralparallelprimitive useful foroffset/index construction andcompaction; illustrativeuses, notclaimedoriginalapplicationprovenance.. Provenance/uncertainty: ReviewedactualnativeCUBsample/historicalCUDAjl source; originalCUBlicenseheader preservedinunchangednativefile. No inferredscientificapplication orhardwarebottleneck.. Domain-neutral examples are illustrative uses, not claimed original application provenance.

## Numerical evidence and implementation ontology

Full59286Int32artifact:3072inputprefix,54×(1024exclusiveprefix+aggregate+16guards). All20CPUchecks andeverytargettrialvsBOTHrefs exact, Int64CPUoracle anddifferenceidentity; fullinputsdevicecopychecked.

Mapping: ThreeCUBstrategies vsoriginalCUDAjlserialscan vsportableBlellochsharedtile; mathematicaloperator/inputs/54pathcount matched, internalalgorithm/transposition/resourceworkdiffers.. Support boundary: Exactsingleblock1024Int32scan,3fixtures/6geometries/all3nativealgorithms; noCUDAjlCUBalgorithmidentityclaim, nooriginalmulti-blockCLI/tails/otherprecisionverification. Per-model implementation records include actual algorithm, layout, mapping, synchronization, resources, pinned versions, allfive samples and hash-linked evidence. The scientific requirement is distinct from the implementation choice.

Accepted bundle `.porting-state/portable_julia/runs/20261018T033128553876Z_blockScan/`. Mean trials 2–5 microseconds: cuda 172.38975000, cuda_jl 2190.24441667, ka 646.67350000, jacc 814.63825000. Timed scope: complete54path_exclusive_scan_suite_per_repeat;3internalrepeats; GPUlaunches+sync, allocation/reset/H2D/oracle/readbackexcluded. Allfivechecks passbothreferences; firsttimingonlyexcluded, nooutlierdrop. Candidate/checkhistory is ineachmodelmanifest; reference/adapterrepairs separatefromportabletranslation. HistoricalCUDA.jlcost remainsunknown, notzero. Current tokenproxies arepartialretainedsourcecharacters/4, input/totalunknown, sharedadaptercountedonce.

## Interpretation for domain scientists

Thisdomain-neutralprimitiveillustratesalgorithm/mappingcost: historicalserialCUDAjl iscorrect, nativeCUB usesoptimizedcooperativevariants, KA/JACC useonecooperativeBlellochscheme. Numericalagreement doesnotmake theirruntime an intrinsicabstraction comparison; sourceCUBalgorithmparameters shouldnotbeinterpretedasimplementedJuliaAPIstrategies.

Confounds: Native3CUBstrategies vsoneJuliaalgorithmpermodel; portablelabelsnotdifferentimplementations; CUDAjloneactivethread vscooperativeKA/JACC; JACCcopies1024templatezerosbeforeoverwritingwithinput; SourceCUBwarp-transpose blockedload/store notliteralportablecoalescedsharedtileload/store; Grid1only; defaultmulti-blockcoincidentwrites/sourceCLIunverified; Allsizes1024, noarbitrarylength/tails; Fiveoneprocessoutertrials/internal3repeats; notstatisticalindependent54experiments; Resourcesunprofiled; no intrinsicframework-onlycausalclaim. RQ1/E1 has configuration-scoped numerical fidelity; RQ2/E2 separates model/phase taxonomy and effort; RQ3/E3 has paired four-model timings; RQ4/E4 links scientific intent to audited implementation traits, with profiler mechanisms pending; RQ5/E5 gives no universal winner. No unsupported feature is inferred from an unfinished translation or an environment error.


## Actual compiler/launch feedback and token coverage

KA passed candidate1/check1. JACC candidate1 compiled but the1024-thread launch rejected with CUDA701 (R1); candidate2 narrows index/stride/N toInt32 and passes the same54paths, geometry and exact acceptance. The precise resource mechanism is unprofiled; no compiler rejection or universal unsupported feature is inferred. Firstfailedcapturecontains no accepted trials. All final20artifacts are bit-identical, including864exactguardvalues pertrial and12288exactinputprefixbytes.

Prospective workflow and model phases are recorded separately; reference and target windows include cached/repeated context and coordination. JACC includes sharedfixture/ontologyannotation whilewaitingforGPUqueue. Model windows overlap the workflow and cannot be added to it or interpreted as cold framework costs. The unchanged historicalCUDAjlbodycost remains unknown.
