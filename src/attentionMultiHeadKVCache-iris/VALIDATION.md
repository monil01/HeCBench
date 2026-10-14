Validation on AMD HIP

Status: ok. Corrected kernel parameter order q,k,v,o. Native HIP and IRIS HIP each pass all four original CUDA cases: FP16/BF16, sequence lengths1/3,32 query heads,4 KV heads,head_dim64,NHD,no positional encoding. Each accuracy=1 at source atol/rtol=1e-2. Copied native loop bounded to CUDA cases; native kernels and sibling defaults untouched. Actual MI100 backend=3 selected; repeat1 validation, default repeat100 retained.

Run with IRIS_ARCHS=HIP from this directory. Full source defaults are preserved; bounded cases are validation coverage only. Attempt history and commands: `.porting-state/porting_logs/attentionMultiHeadKVCache-iris/attempts/attempt_10` (local, not committed).
