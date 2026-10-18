# Quantized barrelwarp errorcontract

{
  "cases": [
    [
      17,
      19,
      0.0,
      0
    ],
    [
      33,
      31,
      0.0009765625,
      1
    ],
    [
      32,
      16,
      6.103515625e-05,
      2
    ],
    [
      19,
      17,
      0.0001220703125,
      3
    ]
  ],
  "inputs": "InterleavedUInt8RGB noise/ramp/checker/constant; all10Float32/nativeInt32propertywords shareexactABI40bytes andnativecalc_shift preprocessingbitcheck",
  "mapping": "16x16 sourcegridfloor(w/16)+1,floor(h/16)+1, deviceglobalpropertypointer all4; historicalJulia byvalueprops alignedseparateP1",
  "outputs": "Every6543RGBbyte+96rawguards includes40bytepropertyguards; UInt32artifactencoding",
  "numerical_rule": "DualGPUatol1byte, independentperchannelderivedFP32intervalstrict: exactbinwhenunique, onlytwoadjacentbinswhenjustified by<=.004705realerror;778ambiguouschannels detailed4cornerweightswitness; zerouncertainboundarymasks; K0identityexact",
  "timing": "Sum4case5launchmeans, contextcomplete, preprocessedproperties/copies/guardreset/readbackoutside",
  "scope": "PredeclaredpositiveKandidentity; sourcequantizedwarp notphysicalcameracalibration/continuousmodelconvergence; historicalstrictCPUbytechecker can differ1owingFMA/truncation"
}

PredeclaredbeforeGPU: native/sourceCPUpropertybits exact, everyRGBbyte mustfall independently derivedperchannelinterval withmaximum .004705continuouserror; exacttruncatedbinexcept778documentedadjacentbinchannels. Fourcorner/weight/errorwitness retained. No uncertainimageboundarymasksallowed infixturedesign. K0identityexact. DualGPUpixelcomparisonatol1doesnotreplace stricteroracle. Allinput/parameterwords and96guards exact. Native sourceunchanged, historicalJulia numericoperatorgenuine; parameterbyvalue->devicepointer mappingalignment isP1, notscientificrepair. Allbinary/textdeclareddependencies hashbound beforefirstmeasurement.
