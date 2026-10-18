# Integer MHC demosaicing

{
  "cases": [
    [
      32,
      10,
      0,
      0
    ],
    [
      35,
      7,
      1,
      1
    ],
    [
      65,
      11,
      2,
      2
    ],
    [
      64,
      10,
      3,
      3
    ],
    [
      64,
      15,
      0,
      4
    ]
  ],
  "patterns": "RGGB/GRBG/GBRG/BGGR, nativekernel acceptsall4; originalCLI selectsRGGBonly",
  "images": "ramp/checker/deterministicnoise/constant123/impulse",
  "layout": "Nativedefault3byteRGBpixel, outputrowpitch4w bytes; wpaddingbyteszero fromreset, notRGBA",
  "filter": "Int32 MHC responses, divisiontrunc0 andsaturateUInt8; reflectexclusiveonefold",
  "mapping": "32x5tile,9x36Int32sharedapron324elements,allpaddedaproncoordinatesinbounds forselectedcases",
  "outputs": "AllRGBbytes+rowpadding, sourceinputreadonly and4guards/case exact; UInt32artifactencodingonly",
  "timing": "Sum5case means of5repetitions CUDAmemset0+ownkernel, contextcomplete;guardresetsetup/readbackoutside",
  "scope": "Namedvaliddimensions/all4kernelpatterns; nooptionalOUTPUT_CHANNELS4build, sensorqualityorallsmallshapesclaim"
}

Predeclaredacceptanceexact0toleranceallRGBandpaddingbytes againstindependentsparseintegerconvolutionoracle, actualsensorchannelpreservation, constantstate123, allinputandguardbytes. Allpaddedapronreflectionscheckedinbounds independentofGPU. Nativegeometryanddefault3channellayout retained, notfabricatedRGBA. SharednativeMHCformula unchanged. Preserve timedruntime memset+kernelpipeline; authoreddevicekernels frameworkown, explicitCUDAhostruntime memoryclearcomponent.
