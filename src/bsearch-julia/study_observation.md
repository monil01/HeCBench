# bsearch: Findthe0-basedbracketindexrwitha[r]<=z<a[r+1] inastrictlyascendingfinitearray

rq_ids: RQ1, RQ2, RQ3, RQ4, RQ5
experiment_ids: E1, E2, E3, E4, E5
configuration_id: bsearch_exact72paths_F32F64_boundarybits_v1

## Scientific intent and fidelity

All4originalnativealgorithmvariants are measured unchanged. HistoricalCUDAjl genuinely implementsBS1only; preservebodyandcomplete3missingvariants insteadoffalselylabelingaliases. KA/JACC preservethefoursearchalgorithms andcooperativeBS4seed semantics.

Scientific task: predecessorsearch/upper_boundminusone for512queries/case,4binarysearchalgorithms. Requirements and precision sensitivity: ExactUInt64indicesand16guards/path; fullinputIEEEbits; everyintervalindependentlychecked. No floatingoutputtolerance.. Community use case: Domain-neutralorderedlookupcanprovide interpolation/bin brackets inscientifictables; illustrativeuse, not anoriginalapplicationclaim.. Provenance/uncertainty: Reviewedactualnativekernel_BS1..4 andhistoricalJuliaexplicitBS1-onlycomment. Namedfixturefinite/sorted/bounded preconditionsmatchsourceintent. No domainapplicationorhardwarebottleneck inferred.. Domain-neutral examples are illustrative uses, not claimed original application provenance.

## Numerical evidence and implementation ontology

50004UInt64artifact:11988losslessIEEEinputwords+72×(512indices+16guards). Independentstd::upper_bound/BaseCPUsearchsortedlast/Pythonbisect plus exacteveryinterval; all5targettrials vsBOTHrefs, inputprefix/guardsbyteexact.

Mapping: All4nativebasic/guardedbitwalk/clampedsentinel/sharedseedalgorithms same256thread2block geometry andinput/querypaths. JACCmode4addsframeworktemplatecopybarrier; hostheterogeneousdispatchdiffers.. Support boundary: All4algorithmvariants/Float32+Float64/9lengths/512finiteboundedqueries+exactguards; notduplicates/NaNs/outsidequeries/arbitrarypartialblocks/default8millionCLI. Per-model implementation records include actual algorithm, layout, mapping, synchronization, resources, pinned versions, allfive samples and hash-linked evidence. The scientific requirement is distinct from the implementation choice.

Accepted bundle `.porting-state/portable_julia/runs/20261018T034251658284Z_bsearch/`. Mean trials 2–5 microseconds: cuda 204.02590000, cuda_jl 251.56720000, ka 413.16035000, jacc 709.61200000. Timed scope: complete72path_binarysearch_suite_per_repeat. Allfivechecks passbothreferences; firsttimingonlyexcluded, nooutlierdrop. Candidate/checkhistory is ineachmodelmanifest; reference/adapterrepairs separatefromportabletranslation. HistoricalCUDA.jlcost remainsunknown, notzero. Current tokenproxies arepartialretainedsourcecharacters/4, input/totalunknown, sharedadaptercountedonce.

## Interpretation for domain scientists

Exactpredecessorqueriesneedthedeclaredcomparisonprecision andboundaries. Independentsortedintervalandinputbitchecks preventsurrogates andcatchFloat64querydowncast. Per-modelresourceandhostdispatch differencesremainperformancecovariates; fourwaynumericalequivalence alone doesnotisolateintrinsicabstractioncost.

Confounds: NewCUDAjlvariants2..4/referencehelpercostseparatefrompreservedhistoricalBS1; Float32/Float64arecomparisons, notfloatingarithmeticthroughput; HeterogeneousJuliaAny[]inputs/runtimeValdispatch andnativevoid*/precisionbranches insidehostinclusiveROI; JACCmode4sharedtemplateinitialization/extrabarrier; Blockaligned512queriesonly; partialcooperativeblocksnotcertified; Finitequerydomaina0<=z<aN; duplicates,NaN,outofrangeunknown; Sharedscalarhelperreusedafterreferencecompletion; modeltokensmarginalordered; No profilermechanism, productionthroughput orcrossvendorclaim. RQ1/E1 has configuration-scoped numerical fidelity; RQ2/E2 separates model/phase taxonomy and effort; RQ3/E3 has paired four-model timings; RQ4/E4 links scientific intent to audited implementation traits, with profiler mechanisms pending; RQ5/E5 gives no universal winner. No unsupported feature is inferred from an unfinished translation or an environment error.


## Actual current workflow and precision witness

Native/CUDAjlreferencecompletion, KA and JACC each passed firstcandidate/check; zero observeddevicecompiler/runtime/numerical failures in this currentworkflow, while historicalattemptcountsremainunknown. All20fullartifactsbitidentical. The referencepreservesgenuineBS1 and adds3missingvariants, ratherthan callingtheoldbaselinefullycomplete. Helperscountonce inreference/shared phase. Prospective modelphasesinclude repeated/cachedcontext/coordination/sharedmetadata; overlapworkflow andarenotcoldintrinsicframeworkcosts.

The independent witness also roundsFloat64queries toFloat32 asacounterfactual, then separates wrongwithin-domainindices fromnewout-of-domainvalues. Actualqueries/acceptanceareunchanged. Thisexposesprecisionchanges that storage labelsalonewouldmiss, withoutclaiming floatingarithmeticthroughput orphysicalapplicationaccuracy.
