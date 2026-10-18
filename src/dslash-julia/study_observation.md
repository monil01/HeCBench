# dslash: Apply the source improved KS fermion Dslash hopping stencil to complex three-component vectors on periodic four-dimensional lattices

rq_ids: RQ1, RQ2, RQ3, RQ4, RQ5
experiment_ids: E1, E2, E3, E4, E5
configuration_id: dslash_complex3_periodic_L4L6_F32_hopping_native_mapping_v1

## Scientific intent and fidelity

Restores historical scalar-only Julia approximation to genuine three-component complexmatrix hopping, preservingoldsource/Makefile. Native kernels unchanged. All4 share sourcecheckerboardneighbor mapping independentlychecked bycoordinate enumeration, full backward adjoints and all threecolor components. Link fixtures are arbitrary complex matrices just as sourceinitializer is not SU3-unitary; identity/constant is an operatorcontrol, not a scientific gauge validation.

Scientific task: Four one-hop and four three-hop complex3x3 link-vector products summed forward, minus backward products with conjugate-transposed links; checkerboard even output sites only.. Requirements and precision sensitivity: All active vectors/backward matrices against independent coordinate/complexFloat64 oracle andbothGPUrefs at2e-5+2e-6|ref|; allinputs exactbytes,600physicalguards and135600unwrittenodd-tail values exact pertrial; constantidentitylink control outputzero.. Community use case: The source header identifies improved KS fermion Dslash and MILC SU3 structures, an operator used in lattice field computations. No physical gaugeensemble/fermioninversion/observable validation is inferred.. Provenance/uncertainty: Native dslash.cu header improved KS fermions; dslash.h MILC SU3 structures and exactneighbor indexing; kernels.cu forward/backward hopping. HistoricalJulia scalar coeff/source shapes lacknative9complexmatrixentries and3complexcomponents; staticS1 diagnosis, notinventedGPUfailure.. Domain-neutral examples are illustrative uses, not claimed original application provenance.

## Numerical evidence and implementation ontology

872062Float64artifactvalues/trial:300134exactactualinput/dimensionprefix +571928fullguardedoutputs/backwardlinks/inputecho; all3colors/bothcomplexcomponents,600physicalguards,135600untouchedodd-tailvalues, bothreferences andindependentcoordinatecomplexCPUoracle. ActualGPUFloat32, encodingFloat64lossless.

Mapping: Sameoneeven-site/lane nativebackwardadjoint+16complexmatvec ordered hopping, sourceone-threadbackwardblocks and256-threadDslashceilgrids; actualFloat32AoSflattening retained. NewCUDAjlrestoresnativealgorithm; portableversions reuse referenceauthoredhelper.. Support boundary: Named3Float32periodiccases fullsourcehopping+checkedbacklinksetup, notphysicalgaugeensemble, normalizedSU3constraint, fullsolver ordefaultCLI.. Per-model implementation records include actual algorithm, layout, mapping, synchronization, resources, pinned versions, allfive samples and hash-linked evidence. The scientific requirement is distinct from the implementation choice.

Accepted bundle `.porting-state/portable_julia/runs/20261018T055026360244Z_dslash/`. Mean trials 2–5 microseconds: cuda 118.13166750, cuda_jl 170.08992000, ka 176.79343500, jacc 190.91222500. Timed scope: sum3 hostinclusive100 synchronized Dslash launches divided100, backward link setup/copies/oracle excluded; five checked trials mean2-5, not full originalCLI. Allfivechecks passbothreferences; firsttimingonlyexcluded, nooutlierdrop. Candidate/checkhistory is ineachmodelmanifest; reference/adapterrepairs separatefromportabletranslation. HistoricalCUDA.jlcost remainsunknown, notzero. Current tokenproxies arepartialretainedsourcecharacters/4, input/totalunknown, sharedadaptercountedonce.

## Interpretation for domain scientists

Preserving a benchmark name is insufficient: the historical scalar approximation omittedmatrix/colorstructure andcheckerboardmapping. Faithfulrestoration ispossible withordinaryKA/JACCdevicecode sharingnative-layoutmath. Scientificprecision andboundarycontracts areexplicit; smallhostinclusiveROIcannotestablish productionlatticebandwidth orrooflineposition. All3JuliaGPUmodels arefirstcurrentcandidates/checks; historiccreationcostunknown.

Confounds: SelectedsmallL4/L6periodiclattices notdefaultL32CLI/fullfermionsolver; Arbitrary complexlinksnotSU3unitarygaugeensemble, sourceinitializeralsonotunitary; Boundeddyadicinputclass happens exactCPUagreement; generalnon-dyadicFMAaccuracyunverified; Backwardadjointsetupfullychecked butoutsideROI asnative, nottimedfullpipeline; SharedreferenceauthoredmaththenKA/JACCreuse, notcoldframeworktokenranking; NativeAoSstructflattenedphysicalrowmajorlayout retained butexactregisterlocalaccumulation/compileroptimizationunprofiled; NativeoriginalMakefile-use_fast_math notadopted; namednvcc-O3defaultFMA studybuild explicitlyarchived, no transcendentaloperationsinselectedbodies; HostinclusiveperlaunchsyncandJACCDictSpec costs included; unprofiledtraffic/cache/occupancy. RQ1/E1 has configuration-scoped numerical fidelity; RQ2/E2 separates model/phase taxonomy and effort; RQ3/E3 has paired four-model timings; RQ4/E4 links scientific intent to audited implementation traits, with profiler mechanisms pending; RQ5/E5 gives no universal winner. No unsupported feature is inferred from an unfinished translation or an environment error.

## Saved-output audit

All20 actual Float32 payloads match the independent dyadic complex oracle exactly as a secondary observed result; original declared tolerance unchanged. All600 physical guards,135600 odd-site unwritten tails and actual inputprefix are exact pertrial. See output_audit.json and scientific_witness.json. The S1 source-only reference disagreement was corrected without changing original native bodies; no full solver or SU3 gauge-field claim.
