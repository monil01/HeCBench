# COBAHH actual-source behavior study, known scientific defect

{
  "cases": [
    [
      33,
      1,
      0
    ],
    [
      257,
      1,
      1
    ],
    [
      129,
      3,
      2
    ]
  ],
  "precision": "Float32 state/coefficients; Int8 refractory flag, artifactFloat32encoding",
  "fixture": "Allneuronsfullyinitialized includingindex0; physiologicalone-step, source-stylepositive/rangestateone-step, symmetricconductances3steps",
  "time_policy": "dt0.0001/t0.01 heldfixed acrossrepeats, lastspikeconstant andflagnotappliedtovoltage, exactlysource",
  "constants": "All33preprocessedactualsourceconstants serializedincludingliteralzeros8/18; expdecayprefactors fixedsharedFloat32",
  "outputs": "All7state/inputfields plusrefractoryflags and48rawguards; exactlastspike/flags/guards",
  "timing": "Sum3 kerneliterationmeans, percaseiterations1/1/3, contextcomplete, reset/readbackoutside",
  "scientific_scope": "Actual truncatedsourcecoefficient stateupdate, NOT faithfulcalibratedHHphysics/networksimulation; known-scientific-defectstratumexcluded",
  "native_initialization_scope": "Originalhostfori=1 leavesindex0undefined; namedinitializedfixtures avoidundefineddatawithoutchangingdevicekernel"
}

Acceptance fixedbeforeGPU: finiteallstates2e-4+2e-4rel againstindependentaffineODE andbothGPUrefs. Refractoryflags,lastspikeinput,guardbytesexact. Expectedphysiologicalm=-0.320847afteroneupdate violatesgatingprobability[0,1] owingliteralzero_lio8; thisscientificdefectisnotignoredorclassifiedframeworkunsupported. Baseline reproducesactualnativebehavior; hypotheticalcorrectedcoefficientsbelongseparatestudybacklog. OriginallargeN/longiterations/undefinedindex0CLI notcertified.
