# MRI-Q Julia ports

This ports both original GPU kernels: squared complex basis magnitude, then the complex Q grid used for convolution-based MRI Hessian multiplication. Original computeQ.cu is included unchanged. It does not implement fullMRI reconstruction or certify the missing external dataset.

From repository root:

```sh
python3 tools/verify_portable_julia.py --config src/mriQ-julia/study_config.json
```

For a fresh all-four series, remove top-level reuse_result and reuse_models from a copied configuration before invoking the runner. Checked historical five-trial bundles are otherwise reused with strictsource/dependency/input checks. Fixtures are deterministic code in study_native.cu and study_common.jl, no binary input asset. Julia1.12.6/CUDA6.2.1/KA0.9.45/JACC1.4.0, localCUDA12.0, nvcc12.0.140, V100GPU6; pinned projects are under src/_julia_{ka,jacc}_env.

Five named (K,X) cases are(1,256),(2,256),(257,512),(1024,256),(1026,512). X alignment is required because the originalQ kernel has no tail bound. OddK only below1024 is tested; odd multi-tile behavior remains unverified and is not a target-framework limitation. Prefix32170 values comprise actual before-input/initial-output/guard readbacks plus every actual uploadedK tile. Remaining22930 values contain fullPhi/Qoutputs, readonly inputs and finalguards. Every55100-value Float32 capture is checked, along with a full independent Float64 directFourier sum, exactzero spatial sample, single-mode modulus, opposite-frequency cancellation and guards.

The ROI sums11upload/launch/completion fragments (5Phi+6Q); Q includes tileupload. Initial allocation/transfers, hostKstructure construction and observations are outside. This differs from uninterrupted originalhostwrapper timing and originaldataset scale. Native fastmath/constant-memory versus Julia math/global-tile paths are confounds. Five trials are checked; discard only firsttime, average lastfour. Error diagnostics use abs(error)/max(abs(reference),1e-4), not unconditionedrelativeerror. See contract.json and paper_observations.json.

JACCfirstcandidate used JACC.Array (host Base.Array), causing a GPUargument-validation rejection; candidate2 uses lowercase JACC.array with unchanged math. Source/log evidence retained. Rootwrongreusekeys initiallyreran references; allactualseries and honestprocesscounts retained in reuse_configuration_erratum.json. Actualtokenwindows include rootcoordination and are excludedfromclean calibration. Historical missingtokenconsumption remainsunknown.

Completion API detail: native uses cudaDeviceSynchronize; Julia uses current-task-stream CUDA.synchronize(). All measured kernels in these harnesses use that stream. Different host wait APIs remain a timing covariate; see completion_scope_review.json. Earlier context-completion wording does not mean identical API calls.
