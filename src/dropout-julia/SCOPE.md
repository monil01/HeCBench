# Exactseeded dropout and originaltimingboundary

{
  "sizes": [
    1028,
    524292
  ],
  "vector_variants": [
    1,
    2,
    4
  ],
  "blocks": [
    512,
    256,
    128
  ],
  "threads": 256,
  "seed": 12345678,
  "offset": 87654321,
  "probabilities": [
    0.25,
    0.5,
    0.75,
    1.0
  ],
  "inputs": "Smallsourceconstant0.1; largevariedsigneddyadicandnegativezero, exactFloat32bits; payload16Balign andNdivisible4",
  "RNG": "ExactPhilox4x32-10counter/subsequence/offsetSTATE1 andFloat32uniformconversion; sourceVEC2alwaysnewuniform4usesfirst2, neverflipunusedbranch",
  "timing": "Sum6original4probabilitylaunchpipelines total(notdivided4), contextcomplete, RNGinitinsidetimedkernels; nohostpremasks",
  "capture_scope": "Actualtimedp4outputs snapshotted BEFORE anyreplay. p1..3 arefulluntimeddeterministicreplays everytrial, notoriginaltimedintermediatearrays. Raw80uintRNGprobeonceperprocess outsidealltiming copiedintoeachcapture explicitlyonce.",
  "oracle": "IndependentPythonintegerPhiloxpackedmask, exactCPUFloat32 input*mask*(1/p), allguards/inputreadonly;80rawuintvendorcurand4probe",
  "mapping": "Native2authoredtemplates/3instances, own2entrydefinitions plusverificationprobe1; packedVecElementV2/V4datatypes, inactiveportablelanesremainatbarrier withRNGworkactiveonly",
  "scope": "All3nativevariants/fixedgrids andactualsecond/fifthloopcontinuation; arbitraryunalignedvector-tailN andlearnedmodelquality unmeasured"
}

Predeclaredacceptance exact0tolerance rawFloat32outputbits (signedzeroincluded), everymaskbit againstindependentintegerPhilox andFloat32scale, allinput/guards. Nativeunchanged. DeviceRNGinitialization insideeverytimedkernel. Originalp1..4samebuffer overwritepipeline preserved; onlyp4capture actualtimedstate; p1..3explicituntimedreplays eachtrial. Auxiliary80rawuintRNGprobe onceperprocess plus6probeguardwords notclaimedfiveindependentruns. 16Bfloat/maskpayloadalign and4-alignedN preconditions. SourceVEC2unusedrandomlanes preserved, noassertidenticalmaskacrossVECvariants. Typedvectorlayout intended; emittedvectorinstructions notyetmeasured, mappingclaimlimitedpendingcompilerinspection.
