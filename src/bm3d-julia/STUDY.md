# BM3D: full hard-threshold denoising pipeline

## Scientific contract

Named fixture:112×40 top-left crop of the supplied noisyPNG red plane, matching the source grayscale channel path; source parameters n19/k8/stack16/T2500/p3/L3D2.7 and actual grayscale variance625. Batch64×12 includes432 valid reference patches. Original1024×768/color conversion/three channels/PNGCLI are not certified. Native source’s rectangle-based horizontal search bounds and packed quantized-distance/offset tie ordering are retained, including asymmetric boundary behavior. No Wiener stage is called by original main.

All seven genuine stages execute: match→gather→2DDCT→stackFWHT/hardthreshold/inverseFWHT/weights→IDCT→atomic overlap aggregation→division. No native code correction. Historical Julia only copied zeros and wrote a black placeholder/PSNR0; it is archived.

## Numerical confirmation

Independent NumPyFloat64 analytical cosine-matrix DCT and Hadamard matrices, integer SSD/key sorting and overlap aggregation provide complete oracles. GPU uses the source factorized8-point arithmetic rather than reusing the oracle equations. Complete actualinput/window/parameters, all matched offsets/counts/weights, all postIDCT groups, both numerator/denominator and every output pixel plus288 typed guard values are captured in all20 trials. Exactdiscrete/input/guard checks; Float32fields atol.003+rtol.0003, UInt8image≤1gray predeclared before timing. Threshold margin in the independent oracle is0.247, and all nonzero-coefficient counts agree exactly. The central mixed-vector tolerance is weaker for integerfields; `audit_saved_outputs.py` independently strengthens every saved dual comparison to the declared fieldwise contract, without retiming or relaxing tolerance. It confirms all20 against CPU and all5 pairs against both GPU references. InputUInt8 values are losslessly serialized as Float32 (bits exactly recoverable), not rawFloat32input storage.

## Scientific counterexample

Against the supplied clean red-plane crop, noisyMSE=11.414955; every captured denoisedMSE is21.284821–21.285491. Thus numerical agreement does not imply improved denoising for this selected crop/source parameters. The native fixed grayscale variance625 is retained; no tuning or favorable-crop selection after seeing quality. This is a specific crop outcome, not proof BM3D generally worsens images or an application-quality certification.

## Implementation ontology and timing

Native matcher shares pixels/differences across16warps and32references (512threads,38,504dynamicsharedbytes); native DCT uses eight lanes/shared tiles and hard filtering64threads with shuffle reduction. All three Julia versions use one worker/reference with private15-key insertion list, one worker/tile with8-tuples and one worker/group with16-value privateFWHT. They preserve the mathematical algorithm and output order but change parallel mapping/local reuse; observed native gaps cannot be assigned purely to abstraction overhead. CUDA.jl uses its own floating atomic_add!; KA/JACC use generic Atomix1.2.1 whose CUDA extension delegates floating add to the same backend intrinsic. All authored variants are framework_device_kernel. JACC explicitly requests zero shared memory and asynchronous stage chaining.

Five in-process checked outer trials follow complete warmup. One timing is the mean of ten complete pipelines; the timed region includes numerator/denominator resets and all seven kernels, with whole-device completion. Other fixture resets, allocations and observations stay outside. Report mean of outer trials2–5. Seconds: {"cuda": 0.0002641844, "cuda_jl": 0.0046563141750000005, "ka": 0.00494795835, "jacc": 0.0044374752500000005}. No new timing series was selected for speed. SourceInteger exactchecks and finite/positive denominator checks accompany all timing rows.

## Effort and provenance

Reference: two device candidates, three process evaluations: host generator syntax rejection beforeJIT; actualdeviceJIT MVector/ntuple inference rejection; staticVal-length correction and pass. One device compiler feedback repair plus one shared host parser repair are separate from the historical semantic restoration. Six authored Julia kernel definitions execute seven stage calls (forward/inverse share a Bool-parameter kernel). Each KA/JACC used one candidate/check and no repairs. Native uses one compiler invocation over adapter+three unchanged source units, not four attempts. Prospective usagewindows includecontext/cache; historical cost remains unknown, reference/shared work separate and source proxies not billed totals.

Reproduce CPU fixtures: `OPENBLAS_NUM_THREADS=1 PYTHONPATH=/tmp/pasc-julia-plot-tools python3 generate_oracle.py` (NumPy2.5.4/Pillow13.0.0); image/generator/fixture hashes recorded. Use `make run` for the four-model study under GPU6 serialization; pinnedJulia1.12.6/CUDAjl6.2.1/KA0.9.45/JACC1.4.0/Atomix1.2.1/localCUDA12.0,sm70. CPU-generated input/oracle binaries are essential reproducible fixtures; measured GPU captures remain local only. Final result: `.porting-state/portable_julia/runs/20261018T040804684889Z_bm3d/result.json`. Paper observation is ready for coordinator integration; worker did not edit paper sections.
