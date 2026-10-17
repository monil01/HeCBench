# fluidSim: scientific and implementation observation

ActualD2Q9benchmarkboundarypolicyisnotautomaticallytextbookclosedwallbounceback:outeredgesdo notpropagate whileinteriorobstaclesdo. Preserveactualoperator,andindependentlychecklocalcollisionmass/momentum,uniformstationarity,andfullpullstreambuffers. Native/Juliaagreementalonewouldnotjustifyclosedwallfluidphysics;no suchclaimmade. HistoricalorientationandrelaxationparameteralignmentsareexplicitP1phases,preventingsilentperformancegainfromthreadmappingchanges.

## Reference audit

HistoricalJuliaD2Q9GPUkernelisgenuine;hostlaunchorientationdiffersactualnative(block256x1vs1x256),andOMEGA1.2Float64differsnativepromotedFloat321.2. PreserveoriginalbytesandGPUmath;declarenativeorientation/parameteralignment+namedheight32scope. ThisisP1mapping/parameteralignment,notestablishedwrongfluidoperator.

## Numerical and performance evidence

Configuration `fluid_d2q9_fp64_256x32_eightsteps_twofields_bothbuffers_nativegeometry_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T212310543764Z_fluidSim/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

One savedcandidate/onecompilerJITcheckeachforalignedCUDA.jl,KA,JACC;noobservedtargetcheckfailure. Originalgenuinekernelmathpreserved;P1hostmapping/parameteralignment documentedseparatelyfromnewtargettranslation. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Observed prospective usage intervals are recorded separately; exclusive billed translation cost is unknown. Final-source proxies remain partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/fluidSim-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.


## Local kinetic conservation and boundary policy

All20artifacts check163864exactFloat64inputslots+294912fullpopulationoutputs/bothpingpongbuffers,withindependentAoScollision/destination-pulloracle maximumerror 3.3306690738754696e-16. Uniformequilibriumstationary,perturbedfieldnondegenerate,and32interiorobstaclesbounceover8steps. Localfluidcollisionmass/momentumconservationwitness [{"perturbed": false, "max_local_fluid_collision_mass_error": 0.0, "max_local_fluid_collision_momentum_error": 0.0, "interior_obstacle_bounce_cell_steps": 0, "initial_total_population": 8192.0, "final_total_population": 8192.0, "global_total_change_observation": 0.0, "max_change_from_initial": 0.0, "minimum_final_population": 0.027777777777777776, "scope": "Localcollisionconservationandactualsource-boundarysemantics;globalclosedwallconservationNOTasserted. Boundarysourcesdonotpropagate;oldunwrittencomponentspersist."}, {"perturbed": true, "max_local_fluid_collision_mass_error": 2.220446049250313e-16, "max_local_fluid_collision_momentum_error": 8.326672684688674e-17, "interior_obstacle_bounce_cell_steps": 256, "initial_total_population": 8704.078125, "final_total_population": 8703.900394992243, "global_total_change_observation": -0.1777300077574182, "max_change_from_initial": 0.03382222002705221, "minimum_final_population": 0.02681520250108507, "scope": "Localcollisionconservationandactualsource-boundarysemantics;globalclosedwallconservationNOTasserted. Boundarysourcesdonotpropagate;oldunwrittencomponentspersist."}]. Globalmasschangeisreportednotassertedzero:actualouteredgesdonotpropagate,butoldunwrittenpopulationcomponentspersist. Noestablishedscientificdefectorunsupportedfeatureisinferredfromtextbookclosedwallassumptions. OriginalJuliaGPUmathretained;nativeorientationandexactpromoted-Float32omegaalignedinexplicitP1referencephase. Nativekernelunchanged,namedheight32notdefault256,andninemacrofieldsinbothstatescheckedat1e-10absolute(all4),stricterthanoldone-sidedpartialcomponent1e-3check.


Prospective rawusage intervals include cached/repeatedcontext and coordination; the parentworkflow contains its childphases and must not be summed with them. Exclusive billed translation cost remains unknown; source-size proxies remain separately labeled.
