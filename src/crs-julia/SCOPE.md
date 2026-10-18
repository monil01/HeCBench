# Cauchy Reed–Solomon actual encoder

{
  "m": [
    1,
    2,
    3,
    4
  ],
  "k": "m..4",
  "w": [
    4,
    5,
    6,
    7,
    8
  ],
  "cases": 100,
  "tasks": [
    1,
    3
  ],
  "per_task_inputwords_perblock": "3w/33w (completewgroups)",
  "data": "UInt64deterministicmixedbits smalltask; larger3tasks zero/allones/mixedbits",
  "matrix": "ExactoriginalGCRSMatrix+galois+jerasure CPU-producedpackedcolumns, independentpolynomialGFverifiedall50",
  "outputs": "Completeeveryparityword, allsourceinputs/matrixreadonlychecked;400serializedinput/outputguards plusmatrixguardsselfchecked",
  "mapping": "Native128threads,worksizefloor128/w*w, originalhostgridceil(size/(4*floor32/w*w)), shared1024bytes, Valm/wtargetspecializations",
  "timing": "Sum100preallocatedkernels-only casepipelines, tasks1/3, fullcontextcompletion; originalH2D/D2Htasktransfers+hostmatrixgeneration excludedall4",
  "scope": "All20nativekernelvariants/all50validmkw atMAX_K4; nooriginalMBsize-throughput/endtoendtransferperformance orGPUdecoderclaim"
}

Predeclaredacceptance exact0tolerance allUInt64paritywords againstindependentcarrylesspolynomialGF oracle plusallinput/matrixreadonly/guards. MDSallk-row-subsetinvertibilityofsystematicCauchymatrix independentlychecks erasurecoding mathematicalstructure; not aGPUdecode test. Completewgroups ensure allread sharedneighborsinitialized includingpartial128threadblocks; inactiveportablelanes participateinbarriers insteadnativeearlyreturn, sameactivegroupalgorithm. No categoricbarrierdefectinoriginalnative asserted. HostGFsetup heldfixtureoutsideROI, source H2D/D2Htaskcopytiming unmeasuredexplicitly.
