# Thirty typed conversions

{
  "sizes": [
    17,
    257
  ],
  "typecodes": {
    "1": "BF16",
    "2": "Float16",
    "3": "Float32",
    "4": "Int32",
    "5": "Int8",
    "6": "UInt8"
  },
  "typed_pairs": [
    [
      1,
      2
    ],
    [
      1,
      3
    ],
    [
      1,
      4
    ],
    [
      1,
      5
    ],
    [
      1,
      6
    ],
    [
      2,
      2
    ],
    [
      2,
      3
    ],
    [
      2,
      4
    ],
    [
      2,
      5
    ],
    [
      2,
      6
    ],
    [
      3,
      3
    ],
    [
      3,
      2
    ],
    [
      3,
      4
    ],
    [
      3,
      5
    ],
    [
      3,
      6
    ],
    [
      4,
      4
    ],
    [
      4,
      3
    ],
    [
      4,
      2
    ],
    [
      4,
      5
    ],
    [
      4,
      6
    ],
    [
      5,
      4
    ],
    [
      5,
      3
    ],
    [
      5,
      2
    ],
    [
      5,
      5
    ],
    [
      5,
      6
    ],
    [
      6,
      4
    ],
    [
      6,
      3
    ],
    [
      6,
      2
    ],
    [
      6,
      5
    ],
    [
      6,
      6
    ]
  ],
  "inputs": "Fullyinitializedfiniteinrangefloatstoallintegers; half/BF16roundingties/subnormals, signedzero, signed/unsigned8extrema, signed32narrowingwrap",
  "outputs": "ExactrawbitpatternalltypedoutputsencodedFloat64; exactsourceinputprefixand4guardspercase(selfchecksource2,serializeddestination2)",
  "timing": "Sum60cases10launchmeans, sourcegridmin(n,256), fullcontextcompletion, allocation/reset/readbackoutside",
  "scope": "All30nativecastpairs, notoriginaluninitializedinputCLI; nooutofrangefloat-to-integer/NaNclaim; narrower signedintegerconversionimplementationpolicy explicitlypinned"
}

Acceptance predeclared: zero-tolerance raw outputbits, inputbits and guards; independent Python IEEEpacking/integertruncation-modulo oracle. Original reads uninitializedallocation, namedfixtures initializeall. No scientificmodel or exceptional/out-of-rangefloatcastclaim. Native percase extra untimedwarmup before each ROI, Julia warms allspecializations before5outertrials; 10 timedcalls equivalent, setup excluded.
