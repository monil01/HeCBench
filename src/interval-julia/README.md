# Interval: full original candidate-root solver, qualified contract

Both original explicit-stack Newton variants, full65,536equations/source64×1024geometry/privatework128/globalresult128, Float64. Source arithmetic currently has directed-rounding intrinsics commentedout: reproduce nearest policy, do notclaim general conservative intervalarithmetic. Allactual roots of the finite original quadratics are covered; source permits extra candidateintervals. `candidate_contract_review.md` preserves explicitpost-observationinterpretation and initialstricterauxiliaryrejection; `verification_contract_revised:true`, no mathematicalnative source repair.

## Reproduction

Use existing pinnedJulia1.12.6/CUDAjl6.2.1/KA0.9.45/JACC1.4.0 environments and CUDA12.0/nvcc/g++17 onthe recordedV100. No system-package changes. Runtimeinputs are sourcebyvalue[Float64(Float32(.01)),4] and exactdyadic threadcoefficients; no external dataset/binaryfixture required. `original_native/` archives originalheaders/main/buildfiles/vendoredBoost copyrights. The complete active devicebody, bothvariants and intervalhelpers are in `study_device.jl`.

```
g++ -O3 -std=c++17 -shared -fPIC src/interval-julia/study_check.cpp -o src/interval-julia/study_check.so
python3 tools/verify_portable_julia.py --config src/interval-julia/study_config_jacc.json
python3 src/interval-julia/audit_exact_roots.py --result .porting-state/portable_julia/runs/20261019T215820483930Z_interval/result.json --raw .porting-state/portable_julia/interval_raw_qualified --output /tmp/interval_reaudit.json
```

Generated checker.so/nativebinaries and multiGBrawcaptures are local rebuildable/outputassets, not committed. For relocation, rewrite absoluteconfig source/env/raw/reuse paths and collect freshseries; archivedlocalrawreferences are not implicitlyportable. Freshnativebuild command is declared in `study_config_native.json`; the native includes unchanged originalGPUheaders. Rebuilds preserve primitivewitness and qualifiedsourcepolicy. Sourceinitialstrictreadiness and freshqualifiedseries are distinct/configsarchived.

Everymodel warms both fullvariants, thenfive independentlyreset checkedtrials. Reportmean2–5 of sumtwokernel launches+contextcompletion; oneinternalrepeat versus original100000 named explicitly. Reset/allocation/actualinputechoes/readbacks/exactrootchecks/canonicalization/rawguardcapture outsideROI. Completeper-equation candidateintervals are sortedwithallmultiplicity/values retained; originalphysicalstack and unusedpoisonedtails kept separately, no raworderidentityclaim.

`primary_root_coverage_receipt.json` binds all20trials/2,621,440equationevaluations and completeactualraw/guardjoins. `all4_exact_root_audit.json` deliberately retains false `strict_enclosure_pass` for the old overstrong everycandidatecontainsroot criterion; primaryrootcoveragepass haszero missingroots, source MAYcontain extrasallowed. Exactpolynomial signs use independentC++128bit and Pythonunboundedintegers, avoidingroundedrootsqrt acceptance. Allsame-input bits, everycount/validendpoint, full128-slotphysicalresultstorage and actualguardbytes retained. No rootexistenceforeachcandidate/universalvalidatednumerics or profiledbottleneck claim.
