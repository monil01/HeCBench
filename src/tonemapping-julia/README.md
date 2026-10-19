# Tone mapping Julia ports

Original GPU kernel is included unchanged from tonemapping-cuda/kernels.cu; see originallicense notice in LICENSE.source. Eachtargethas oneframeworkalgorithmkernel andoneuntimedscalarobserver. device_math.jl is asharedtranslatedpuredevicefunction authoredinCUDAjlphase, genuinelycompiledbyCUDAjl/KA/JACC, notnativekerneldelegation.

```sh
python3 tools/verify_portable_julia.py --config src/tonemapping-julia/study_config.json
```

Runfromroot. Fornewallfourmeasurements remove reuse_result/reuse_models fromacopiedconfig; otherwise strictlysource-boundfive-trialreferencebundles are reused. Fixturesgeneratedincode, noexternalbinaryassets. PinnedJulia1.12.6/CUDA6.2.1/KA0.9.45/JACC1.4.0/localCUDA12/nvcc12.0.140/V100GPU6.

Each1093708Float32artifact includes546868exactactualinput/initialoutput/scalar/guardprefix plus546840fulloutput/readonly/finalguardvalues. Float64physicalimageoracle, uniformgrayclosedform, exactalpha, positivefinitecolor, guards. Seecontract.json forpositiveRGB/alignedRGBApreconditions; black0/0sourcebehavior isoutsideacceptedcontract andnotframeworklimitation.

Fivecheckedtrials, meanlastfour; ROI sumsfouroriginal16x16kernel+completionregions. No externalHDRparser/defaultimage/10,000repeat/fullapplicationclaim. Additionalpostmeasurement originalFloat32CPU signedmean1e-6check passesall20savedcaptures; source_criterion_audit.json pins exactextractedblocks/source/result/artifacts. Its aggregateweakness isexplicit; primaryfullpointwiseFloat64checks unchanged.

audit_original_source_criterion.cpp is generated fromunchangedmain source reference/metricblocks; source_criterion_audit.json retains theirhashes andcompilercommand. ReexecutecompiledCPUauditor withfullsavedcapturepaths onlywhenavailable. RawGPUcaptures local, notGit/Overleaf.
