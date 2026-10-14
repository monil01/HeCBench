# Selective scan, IRIS dmem

Retains the original base recurrence and all eight VLLM-style scan template
specializations, including the original HIP device-side hipCUB BlockScan helper.
No host vendor operation is replaced. A flat dmem kernel ABI reconstructs the
original parameter struct inside the GPU wrapper. Static shared storage has
the exact original trait-defined size; the runtime has no dynamic shared-size
parameter. Grid axes preserve the original base and scan launches.

All host GPU storage uses iris::DMem. State resets update dmem; complete output
and state arrays are explicitly flushed. Synchronous task/flush overhead and
host-backed carry allocation differ from native timing; no throughput parity
claim. CPU reference, seed19937, input formulas and processing order unchanged.

make/run defaults remain B32,D2048,N8,L1024 and 1000 repetitions. Opt-in
EXTRA_CFLAGS=-DSSM_BATCH=2 -DSSM_DIM=16 -DSSM_SEQLEN=512 changes only validation
dimensions; both models use identical documented input adapters. The original
SSM implementation does not synchronize separate chunk blocks. That original
behavior is preserved, not repaired or replaced by serialized chunk execution.

Validation on HIP covers single even/odd chunks, two repetitions and two-chunk
B2,D2048,L1024. All12,877,218 output/state float values matched native HIP
byte-for-byte, including final output/state after the benchmark repeat loops;
original independent CPU reference checks passed in both models.
Full production defaults and NVIDIA execution are not claimed validated.
SSM_VERIFY_PREFIX optionally writes full output/state arrays for validation.
