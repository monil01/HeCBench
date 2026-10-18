# BatchedGEMM predeclared vendor-interoperability study

{
  "upper_leading_dimension": 17,
  "sizes": [
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17
  ],
  "batches": [
    3,
    17
  ],
  "types": [
    "Float16",
    "Float32",
    "Float64"
  ],
  "fixtures": [
    "Signedexactdyadic",
    "Signedroundedrational"
  ],
  "alpha": 1,
  "beta": 0,
  "transpose": [
    "N",
    "N"
  ],
  "gemm_n": 1,
  "timing": "Sum192 size/type/batch/fixture points, each mean10calls rep1..10 with rep0excluded; contextcomplete eachcall. Allocation/inputsetup/readbackoutside. Fiveoutertrials discardonlyfirstoutertiming.",
  "output": "Full17*batchoutputeachsize including exactinactivepadding andouterguardvalues plusstatus; allinputandpointerlistscheckedbitexact",
  "input_encoding": "Float64serializedactualFloat16/Float32/Float64values exactlyrepresentable inFloat64; exactbyteprefix acrossall4",
  "independent_acceptance": "Float64fsum dots, boundgamma_2m(typeunit)*sumabsproducts+4unit*absoracle, paddingandguardsbitexact",
  "library": "Pinned+preloaded cuBLAS/Lt12.0.2.224 all4, actualruntimepath/version printed; explicitJuliahandletaskstreambinding/contextcompletion",
  "scope": "Zeroauthoredkernels; KA/JACCframeworkallocation withCUDAvendorinterop. Originalupper100/batch25000CLIunmeasured."
}

Cross-model actualoutputs/inputprefixexact atol=rtol=0. Independent numerical acceptance uses standardroundoffdotbound bystorage/APIprecision; internalcuBLASaccumulationprecisionnotinferred. Allguards/padding/input/pointerlistbytescheckedexact. H/S/D cases retained; original fullCLI notcertified. Runtime-librarypath/version/hash evidence mandatory, genericlinkflag alone insufficient.
