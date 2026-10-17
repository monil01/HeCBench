# axhelm: scientific and implementation observation

Axhelmrequirespreservingtheweightedtensoroperatorandfusedthreefieldmapping;unrelatedhistoricalaliashasnomeaningfulscientificresult. SharedGaussLegendrebarycentricDandpositiveSPDweightedgeometrysupportindependentpolynomialderivative/energy/constantfieldwitnesses. Endtoendnativeallocation/transfer/freewindowisretained,soitsperformancecannotbeinterpretedaskernel-onlyrooflineposition. ElementlocalapplicationdoesnotestablishglobalPDEboundaryconditions/solverconvergence.

## Reference audit

HistoricalJuliaentrypointincludesunrelatedmcprbenchmark,notHelmholtz. GenuineS1repair preservesnative8x8scalarandfused3kernels,8privatezplanes,sharedD/Q/fluxplanes,distinctscalar/fusedaccumulationandbarriercounts.

## Numerical and performance evidence

Configuration `axhelm_degree7_gauss_spd7elements_scalar_fused3_e2e_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T200940213456Z_axhelm/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

HistoricalunrelatedreferenceS1repairseparatefromtargettranslation. ReferenceandKApassfirstdevicecandidate/check;JACCoutcomeindependentlyrecorded. Namedscientificfixture/basis/oracleconstructionandend-to-endharnessworkarepartofsharedworkflow,notfree target-costzeros. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Observed prospective usage intervals are recorded separately; exclusive billed translation cost is unknown. Final-source proxies remain partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/axhelm-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.


## Scientific witness and timing scope

All 14,336 nodal values in each of all20 artifacts pass the independent Float64 direct tensor-gradient/flux/transpose oracle. The actual Float32 differentiation matrix reproduces degree0–7 derivatives within5.96e-7; underlying Float64 basis error4.22e-15. Positive Gauss weights and strictly diagonally dominant symmetric diffusion tensor establish positive local energy; constant-field diffusion is small but not exactlyzero afterFloat32basisrounding. Scalar and fused-three-independent-field mappings preserve native8x8threads and private8planes with1024/2560Bshared. Originalallocation/H2D/100repeats/D2H/free region is timed, CUDAmemorypoolnone; allocatorAPI differences remain declared, so these are not kernel-onlyroofline rates. JACC.shared uses a view of existing readonlygeometry for its extra1024/2560Binitialcopy; noGPUtemplateallocation is hidden outside the measuredregion. Named7elements, not8000; no global boundarycondition/solverconvergenceclaim.


Prospective rawusage intervals include cached/repeatedcontext and coordination; the parentworkflow contains its childphases and must not be summed with them. Exclusive billed translation cost remains unknown; source-size proxies remain separately labeled.
