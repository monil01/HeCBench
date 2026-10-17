# Reference blocker: FP8/BF16 QKV preparation

Native `FP8TrtAddQKVBiasKernel` uses FP8 E4M3 storage and BF16x2 multiply/add intermediates before requantization and axis permutation. The unchanged source fails CUDA12.0 compilation for available V100sm70 and succeeds for sm80. `required_bf16_probe.cu` preserves the smallest required BF16x2 multiply; it fails sm70 compile, compiles sm80, then returns CUDAerror209 (no kernel image) when run on V100. See hashed diagnostic summary in `diagnostic_evidence.json`; complete raw logs remain local STATE.

Reproduce compile with `nvcc -O3 -std=c++17 -arch=sm_70 required_bf16_probe.cu -o probe70` and `-arch=sm_80 ... -o probe80`; execute probe80 only under studyGPU6 lock. Historical CUDA.jl entry point aliases unrelated mcpr (S1). Neither portable framework has been translated or compiled here; this is a native hardware/reference blocker, not evidence of unsupported KA/JACC APIs. No precision emulation is substituted.

Resume on BF16-capable sm80+ hardware, repair genuine CUDA.jl reference with separate effort, and then translate KA followed by JACC. No correctness or timing result is claimed.
