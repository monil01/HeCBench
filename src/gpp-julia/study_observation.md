# gpp: Evaluate a generalized plasmon-pole self-energy summation at three frequencies

rq_ids: RQ1, RQ2, RQ3, RQ4, RQ5
experiment_ids: E1, E2, E3, E4, E5
configuration_id: gpp_F64_four_bounded_uniform_varied_gridstride_zero_cases_v1

## Scientific intent and fidelity

Preserve unchanged native AoS and historical genuine CUDA.jl SoA operators. New independent full-frequency checker replaces original first-frequency one-sided host checker. Portable versions keep per-thread accumulators/grid-stride loops and six Float64 atomics.

Scientific task: Sum complex rational pole factors times inverse-dielectric and conjugated band-amplitude products over bands, owned reciprocal vectors and plane waves; six scalar Float64 atomic reductions.. Requirements and precision sensitivity: Full six output scalars against independent Decimal80 allterm oracle at predeclared 1e-11+1e-11|gold|; all input bits/guards exact, both GPU references checked on every trial.. Community use case: Self-energy miniapplication extracted from BerkeleyGW electronic excited-state/material calculations. This is a kernel operator study, not a physical quasiparticle prediction or complete BerkeleyGW validation.. Provenance/uncertainty: Repository README gpp description and MIT2019NERSC native sources. Primary background: Vergara Larrea et al., CUG2019 Experiences Porting Mini-applications..., https://cug.org/proceedings/cug2019_proceedings/includes/files/pap137s2-file1.pdf sections introduction/GPP identify BerkeleyGW General Plasmon Pole self-energy summation; no new material-science prediction claim.. Domain-neutral examples are illustrative uses, not claimed original application provenance.

## Numerical evidence and implementation ontology

All20 capture 18809 Float64 words/trial: 8955 exact input/metadata prefix and 9854 full guarded outputs/inputecho. 24 outputscalars, 896physicalguards and8934 actual inputpayloadvalues/trial checked independently; exactzero control, finite denominator controls. Decimal80 allterm oracle and both GPU refs; no checksum substitute.

Mapping: Same per-thread band/ownedG/plane-wave gridstrides, 32thread2D grid and six output atomics. Historical genuine SoA complex storage and component division vs native AoS/reciprocal multiply remain numerical/layout adaptations; KA/JACC use same arithmetic helper transformed from historical CUDAjl body.. Support boundary: Four named nonsingular small Float64 finite cases, three full complex frequencies. Source unusednvband remains unused; full application and singular/resonant/large configurations unverified.. Per-model implementation records include actual algorithm, layout, mapping, synchronization, resources, pinned versions, allfive samples and hash-linked evidence. The scientific requirement is distinct from the implementation choice.

Accepted bundle `.porting-state/portable_julia/runs/20261018T071302525269Z_gpp/`. Mean trials 2–5 microseconds: cuda 65.56550000, cuda_jl 143.69400000, ka 175.10175000, jacc 245.62000000. Timed scope: Sum4hostinclusive singlekernel launch+synchronize intervals. Resetoutputs/inputcopies/JITwarmup/readback/oracle/serialization outside.5checkedinprocess trials, meanlast4, alltimingsretained; no original10runCLItime claim.. Allfivechecks passbothreferences; firsttimingonlyexcluded, nooutlierdrop. Candidate/checkhistory is ineachmodelmanifest; reference/adapterrepairs separatefromportabletranslation. HistoricalCUDA.jlcost remainsunknown, notzero. Current tokenproxies arepartialretainedsourcecharacters/4, input/totalunknown, sharedadaptercountedonce.

## Interpretation for domain scientists

For electronic-structure users this tests the self-energy summation kernel and safe nonsingular selected data, not manybody-model accuracy. Layout and division adaptations confound abstraction comparisons; atomic-order differences are permitted only inside declared full-output error bounds. No framework limitation or measured bottleneck inferred.

Confounds: Native AoS complex vs Julia SoA and reciprocalmultiply vs directdivision; Small finite selected kernels, not original fullCLI/physical selfenergy prediction; Float64 atomic ordering may vary between trials; tolerant numeric agreement not exact raw bitwise equality; One shared scientific helper authored during KA translation and reused by JACC; ordered workflow marginal cost; Host-inclusive sum4 short kernel dispatches; unprofiled kernel/launch/resource bottlenecks; Original native/CUDAjl first-frequency one-sided host checkers invalid as general accuracy certificates; unchanged operator independently checked. RQ1/E1 has configuration-scoped numerical fidelity; RQ2/E2 separates model/phase taxonomy and effort; RQ3/E3 has paired four-model timings; RQ4/E4 links scientific intent to audited implementation traits, with profiler mechanisms pending; RQ5/E5 gives no universal winner. No unsupported feature is inferred from an unfinished translation or an environment error.

## Scientific and effort qualification

Electronic-structure selfenergy summation, not generic statistics. Historical genuineCUDAjl GPUbody unchanged; one currentexistingcandidatecheck, zero newreferenceGPUtranslations, creationhistoryunknown. Sharedhelper firstauthoredKAphase thenJACCreuse. Originalhostchecker staticfalseacceptance replaced with Decimal80 fulloracle, no nativeGPUbodyrepair/failure. NativeAoS vsJuliaSoA andreciprocalmultiply vscomponentdivision confoundframeworkcausality. Atomicrawsamples retained; tolerantfulloutput, notproperty-only orbitwiseagreement.
