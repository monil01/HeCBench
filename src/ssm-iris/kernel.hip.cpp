#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
static constexpr int kDstate=8;
__device__ __forceinline__ float softplus_f(float x) {
    return (x > 20.f) ? x : __logf(1.f + __expf(x));
}
__device__ __forceinline__ float silu_f(float x) {
    return x / (1.f + __expf(-x));
}

extern "C" __global__ void selective_scan_base(
    const float* __restrict__ u,
    const float* __restrict__ delta,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ C,
    const float* __restrict__ D_vec,
    const float* __restrict__ delta_bias,
    const float* __restrict__ z,         // may be nullptr
    bool   delta_softplus,
    int    batch, int dim, int dstate, int seqlen,
    float* __restrict__ ssm_states,
    float* __restrict__ y
) {
    int b = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= batch || d >= dim) return;

    float db = delta_bias[d];
    float Dv = D_vec[d];
    float h[kDstate];
    for (int n = 0; n < dstate; ++n)
       h[n] = ssm_states[b*dim*dstate + d*dstate + n];

    for (int l = 0; l < seqlen; ++l) {
        float u_v  = u    [b*dim*seqlen + d*seqlen + l];
        float dt   = delta[b*dim*seqlen + d*seqlen + l] + db;
        if (delta_softplus) dt = softplus_f(dt);

        float yv = Dv * u_v;
        for (int n = 0; n < dstate; ++n) {
            float dA = __expf(dt * A[d*dstate + n]);
            float dB = dt * B[b*dstate*seqlen + n*seqlen + l];
            h[n] = dA * h[n] + dB * u_v;
            yv  += C[b*dstate*seqlen + n*seqlen + l] * h[n];
        }
        if (z) yv *= silu_f(z[b*dim*seqlen + d*seqlen + l]);
        y[b*dim*seqlen + d*seqlen + l] = yv;
    }
    for (int n = 0; n < dstate; ++n)
      ssm_states[b*dim*dstate + d*dstate + n] = h[n];
}

template<int kNThreads_, int kNItems_, int kDstate_, bool kHasZ_,
         bool kDeltaSoftplus_, bool kIsEvenLen_>
struct SSMFwdKernelTraits {
    static constexpr int kNThreads      = kNThreads_;
    static constexpr int kNItems        = kNItems_;
    static constexpr int kDstate        = kDstate_;
    static constexpr bool kHasZ         = kHasZ_;
    static constexpr bool kDeltaSoftplus = kDeltaSoftplus_;
    static constexpr bool kIsEvenLen    = kIsEvenLen_;
    static constexpr int kChunkSize     = kNThreads * kNItems;

    // Associative scan element: (a, b) representing  h_t = a*h_{prev} + b
    struct ScanElement {
        float a;   // multiplicative factor (= A_bar)
        float b;   // additive   factor   (= B_bar * u)
    };

    // CUB associative operator  ∘
    struct ScanOp {
        __device__ __forceinline__ ScanElement
        operator()(const ScanElement& lhs, const ScanElement& rhs) const {
            // rhs is "later" in time:  h = rhs.a*(lhs.a*h0 + lhs.b) + rhs.b
            return { rhs.a * lhs.a,  rhs.a * lhs.b + rhs.b };
        }
    };

    using BlockScanT = hipcub::BlockScan<ScanElement, kNThreads,
                                      hipcub::BLOCK_SCAN_WARP_SCANS>;

    // Shared memory:  one BlockScan workspace + tile buffers
    // running_prefix_smem[n]: broadcast slot so the thread that owns the last
    // valid sequence position within this chunk can share its carry with:
    //   (a) all other threads for y accumulation, and
    //   (b) threads 0..kDstate-1 for the final ssm_states writeback.
    // last_valid_thread: index of the thread that owns seq position
    //   min(chunk_end, seqlen) - 1  (needed for non-even-length chunks).
    struct SharedMemory {
        typename BlockScanT::TempStorage scan_storage;
        float u_tile    [kChunkSize];
        float delta_tile[kChunkSize];
        // B and C: (kDstate, kChunkSize) – laid out [state_dim][time]
        float B_tile    [kDstate * kChunkSize];
        float C_tile    [kDstate * kChunkSize];
        // Per-state carry broadcast slot (written by last-valid thread, read by all)
        float carry_a   [kDstate];
        float carry_b   [kDstate];
        int   last_valid_thread;  // thread index owning the last valid seq element
        int   last_valid_item;    // item index within that thread
    };

    static constexpr int kSmemSize = sizeof(SharedMemory);
};

struct SSMParamsBase {
    int    batch, dim, seqlen, dstate, n_chunks;
    // Pointers
    float *u_ptr, *delta_ptr, *A_ptr, *B_ptr, *C_ptr,
          *D_ptr, *delta_bias_ptr, *z_ptr,
          *ssm_states_ptr, *out_ptr;
    // Inter-chunk carry buffers (global memory, shape: batch*dim * n_chunks * dstate)
    // carry_a[chunk] and carry_b[chunk] store the ScanElement after each chunk.
    // chunk 0 initialises from ssm_states; chunks 1..n_chunks-1 read from here.
    float *running_prefix_a;   // multiplicative carry  (a field of ScanElement)
    float *running_prefix_b;   // additive    carry     (b field — the actual h)
    // Strides (elements, not bytes) — mirrors vLLM's explicit stride fields
    int u_batch_stride,      u_d_stride;        // for u and delta (same shape)
    int B_batch_stride,      B_dstate_stride;   // for B and C
    int states_batch_stride, states_d_stride;
    // Stride into the running_prefix buffers: [bd_idx * dstate * n_chunks + n * n_chunks + chunk]
    // (bd_idx = b*dim+d)
};

template<typename Traits>
__device__ __forceinline__ void
selective_scan_vllm_kernel(SSMParamsBase params)
{
    constexpr int kNThreads  = Traits::kNThreads;
    constexpr int kNItems    = Traits::kNItems;
    constexpr int kChunkSize = Traits::kChunkSize;
    constexpr int kDstate    = Traits::kDstate;

    // Block → (batch, dim) mapping
    // Grid: (batch * dim,  n_chunks)
    const int bd_idx  = blockIdx.x;              // linearised batch*dim index
    const int chunk   = blockIdx.y;              // which chunk of L

    const int b = bd_idx / params.dim;
    const int d = bd_idx % params.dim;

    const int chunk_start = chunk * kChunkSize;  // first seq index this block

    // Pointers to this row's data
    const float* u_row     = params.u_ptr
                             + b * params.u_batch_stride
                             + d * params.u_d_stride;
    const float* delta_row = params.delta_ptr
                             + b * params.u_batch_stride
                             + d * params.u_d_stride;
    const float* z_row     = (Traits::kHasZ) ?
                             params.z_ptr + b * params.u_batch_stride
                                          + d * params.u_d_stride
                             : nullptr;
    float*       out_row   = params.out_ptr
                             + b * params.u_batch_stride
                             + d * params.u_d_stride;

    __shared__ typename Traits::SharedMemory smem;

    // Compute last valid thread/item within this chunk
    // "last valid" = the thread that processes seq position (last_seq_idx):
    //   even-len path: all positions in [chunk_start, chunk_start+CHUNK) valid
    //                  → last thread (kNThreads-1), last item (kNItems-1)
    //   odd-len  path: last valid seq idx = min(chunk_start+CHUNK, seqlen) - 1
    if (threadIdx.x == 0) {
        if (Traits::kIsEvenLen) {
            smem.last_valid_thread = kNThreads - 1;
            smem.last_valid_item   = kNItems   - 1;
        } else {
            int last_seq = min(chunk_start + kChunkSize, params.seqlen) - 1;
            int local    = last_seq - chunk_start;          // 0-based within chunk
            smem.last_valid_thread = local / kNItems;
            smem.last_valid_item   = local % kNItems;
        }
    }
    __syncthreads();

    const int last_thr  = smem.last_valid_thread;
    const int last_item = smem.last_valid_item;

    // Load u and delta tiles into shared memory (coalesced)
    // Each thread loads kNItems consecutive elements.
    int base = chunk_start + threadIdx.x * kNItems;
    #pragma unroll
    for (int i = 0; i < kNItems; ++i) {
        int l = base + i;
        bool valid = Traits::kIsEvenLen || (l < params.seqlen);
        float u_v  = valid ? u_row    [l] : 0.f;
        float dt_v = valid ? delta_row[l] : 0.f;
        dt_v += params.delta_bias_ptr[d];
        if (Traits::kDeltaSoftplus) dt_v = softplus_f(dt_v);
        smem.u_tile    [threadIdx.x * kNItems + i] = u_v;
        smem.delta_tile[threadIdx.x * kNItems + i] = dt_v;
    }
    __syncthreads();

    // Per-state-dimension parallel scan
    float D_val = params.D_ptr[d];

    // Initialise output accumulator to D * u (skip z for now)
    float y_tile[kNItems];
    #pragma unroll
    for (int i = 0; i < kNItems; ++i) {
        int li = threadIdx.x * kNItems + i;
        y_tile[i] = D_val * smem.u_tile[li];
    }

    // Inter-chunk carry index layout:
    //   running_prefix_{a,b}[ bd_idx * kDstate * n_chunks + n * n_chunks + chunk ]
    // chunk==0: carry initialised from ssm_states (initial hidden state h0).
    // chunk>0 : carry loaded from the previous chunk's global-memory entry.
    // Within each state-dim iteration the carry is broadcast via smem.carry_{a,b}[n].

    // Iterate over each SSM state dimension independently
    for (int n = 0; n < kDstate; ++n) {

        // Load carry for this (bd_idx, n, chunk)
        // All threads need the same carry scalar; thread 0 loads it and
        // broadcasts via smem.carry_{a,b}[n].
        using SE = typename Traits::ScanElement;
        if (threadIdx.x == 0) {
            SE carry;
            int carry_base = bd_idx * kDstate * params.n_chunks
                           + n      * params.n_chunks;
            if (chunk == 0) {
                // Encode initial state h0 as ScanElement{1, h0}:
                //   combined.b = prefix.a * 1 * h0 + prefix.b
                float h0 = params.ssm_states_ptr[b * params.states_batch_stride
                                                + d * params.states_d_stride + n];
                carry.a = 1.f;
                carry.b = h0;
            } else {
                carry.a = params.running_prefix_a[carry_base + (chunk - 1)];
                carry.b = params.running_prefix_b[carry_base + (chunk - 1)];
            }
            smem.carry_a[n] = carry.a;
            smem.carry_b[n] = carry.b;
        }

        // Load B and C tiles for state n
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
            int l = base + i;
            bool valid = Traits::kIsEvenLen || (l < params.seqlen);
            smem.B_tile[n * kChunkSize + threadIdx.x * kNItems + i] =
                valid ? params.B_ptr[b * params.B_batch_stride
                                    + n * params.B_dstate_stride + l] : 0.f;
            smem.C_tile[n * kChunkSize + threadIdx.x * kNItems + i] =
                valid ? params.C_ptr[b * params.B_batch_stride
                                    + n * params.B_dstate_stride + l] : 0.f;
        }
        __syncthreads();   // carry_a/b[n] and B/C tiles are ready

        SE carry = { smem.carry_a[n], smem.carry_b[n] };

        // Build per-thread ScanElements (one per item)
        SE thread_data[kNItems];
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
            int li    = threadIdx.x * kNItems + i;
            float dt  = smem.delta_tile[li];
            float u_v = smem.u_tile[li];
            float Bn  = smem.B_tile[n * kChunkSize + li];
            float a_bar = __expf(dt * params.A_ptr[d * kDstate + n]);
            float b_bar = dt * Bn * u_v;
            thread_data[i].a = a_bar;
            thread_data[i].b = b_bar;
        }

        // Inclusive prefix scan across the block
        using BlockScanT = typename Traits::BlockScanT;
        using ScanOp     = typename Traits::ScanOp;
        SE prefix_out[kNItems];
        BlockScanT(smem.scan_storage).InclusiveScan(
            thread_data, prefix_out, ScanOp{}
        );
        __syncthreads();

        // Incorporate carry and accumulate y
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
            // combined = carry ∘ prefix_out[i]
            SE combined;
            combined.a = prefix_out[i].a * carry.a;
            combined.b = prefix_out[i].a * carry.b + prefix_out[i].b;
            float h_t = combined.b;
            int li = threadIdx.x * kNItems + i;
            y_tile[i] += smem.C_tile[n * kChunkSize + li] * h_t;
        }

        // ── Write new carry to global memory and smem broadcast slot ─────────
        // The carry after this chunk = the combined ScanElement at the LAST
        // VALID sequence position within the chunk.
        // For even-length chunks this is always the last thread's last item.
        // For odd-length (padded) chunks this may be an earlier thread/item.
        // Using smem.last_valid_thread / last_valid_item computed above.
        if (threadIdx.x == last_thr) {
            SE combined;
            combined.a = prefix_out[last_item].a * carry.a;
            combined.b = prefix_out[last_item].a * carry.b + prefix_out[last_item].b;

            // Write to global memory for subsequent chunks
            int carry_idx = bd_idx * kDstate * params.n_chunks
                          + n      * params.n_chunks
                          + chunk;
            params.running_prefix_a[carry_idx] = combined.a;
            params.running_prefix_b[carry_idx] = combined.b;

            // Also broadcast into smem so threads 0..kDstate-1 can read
            // it for the final state writeback (avoids a second global read).
            smem.carry_a[n] = combined.a;
            smem.carry_b[n] = combined.b;
        }
        __syncthreads();
    }
    // After the n-loop, smem.carry_{a,b}[n] holds the end-of-chunk hidden
    // state for every n, written by last_thr and visible to all threads.

    // Apply optional SiLU z-gate and write outputs
    #pragma unroll
    for (int i = 0; i < kNItems; ++i) {
        int l = base + i;
        bool valid = Traits::kIsEvenLen || (l < params.seqlen);
        if (!valid) continue;

        float yv = y_tile[i];
        if (Traits::kHasZ) yv *= silu_f(z_row[l]);
        out_row[l] = yv;
    }

    // Write final SSM state back to global memory (last chunk only)
    // Thread n writes state dimension n, reading from smem.carry_b[n] which
    // holds h_L = the hidden state after the last valid sequence position.
    if (chunk == params.n_chunks - 1 && threadIdx.x < kDstate) {
        int n = threadIdx.x;
        params.ssm_states_ptr[b * params.states_batch_stride
                             + d * params.states_d_stride + n]
            = smem.carry_b[n];
    }
}

static constexpr int kNT = 128;
static constexpr int kNI = 4;


#define SSM_POINTERS float *u, float *delta, float *a, float *b, float *c, float *d, float *dbias, float *z, float *states, float *out, float *carry_a, float *carry_b
#define SSM_SCALARS int batch, int dim, int seqlen, int dstate, int chunks
#define SSM_EXPORT(NAME,HZ,SP,EL) \
extern "C" __global__ void NAME(SSM_POINTERS, SSM_SCALARS) { \
  SSMParamsBase p{}; \
  p.batch=batch; p.dim=dim; p.seqlen=seqlen; p.dstate=dstate; p.n_chunks=chunks; \
  p.u_ptr=u; p.delta_ptr=delta; p.A_ptr=a; p.B_ptr=b; p.C_ptr=c; p.D_ptr=d; \
  p.delta_bias_ptr=dbias; p.z_ptr=HZ?z:nullptr; p.ssm_states_ptr=states; p.out_ptr=out; \
  p.running_prefix_a=carry_a; p.running_prefix_b=carry_b; \
  p.u_batch_stride=dim*seqlen; p.u_d_stride=seqlen; \
  p.B_batch_stride=dstate*seqlen; p.B_dstate_stride=seqlen; \
  p.states_batch_stride=dim*dstate; p.states_d_stride=dstate; \
  selective_scan_vllm_kernel<SSMFwdKernelTraits<kNT,kNI,kDstate,HZ,SP,EL>>(p); \
}
SSM_EXPORT(ssm_111,true,true,true)
SSM_EXPORT(ssm_110,true,true,false)
SSM_EXPORT(ssm_101,true,false,true)
SSM_EXPORT(ssm_100,true,false,false)
SSM_EXPORT(ssm_011,false,true,true)
SSM_EXPORT(ssm_010,false,true,false)
SSM_EXPORT(ssm_001,false,false,true)
SSM_EXPORT(ssm_000,false,false,false)
