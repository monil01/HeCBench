using CUDA
using Printf
using Random

const THREADS = 256

function scaled_masked_softmax_kernel!(dst, src, mask, scale::Float32,
                                       rows::Int64, key_seq_len::Int32,
                                       query_seq_len::Int32, pad_batches::Int32, heads::Int32)
    row = blockIdx().x
    tid = threadIdx().x
    stride = blockDim().x
    if row <= rows
        q = Int32((row - Int32(1)) % query_seq_len + Int32(1))
        batch_index = (row - Int32(1)) ÷ (Int64(heads) * Int64(query_seq_len))
        mask_row = pad_batches == Int32(1) ? q : batch_index * Int64(query_seq_len) + q
        maxval = -10000.0f0
        for col in tid:stride:key_seq_len
            midx = (Int64(mask_row) - 1) * Int64(key_seq_len) + Int64(col)
            idx = (Int64(row) - 1) * Int64(key_seq_len) + Int64(col)
            m = @inbounds mask[midx]
            v = m != UInt8(1) ? (Float32(@inbounds(src[idx])) * scale) : -10000.0f0
            maxval = max(maxval, v)
        end
        sh = CUDA.@cuStaticSharedMem(Float32, THREADS)
        sh[tid] = maxval
        sync_threads()
        offset = blockDim().x ÷ Int32(2)
        while offset > 0
            if tid <= offset
                sh[tid] = max(sh[tid], sh[tid + offset])
            end
            sync_threads()
            offset ÷= Int32(2)
        end
        maxval = sh[1]
        sync_threads() # all warps consume max before shared buffer reuse
        full_mask = maxval == -10000.0f0

        sumval = 0.0f0
        for col in tid:stride:key_seq_len
            midx = (Int64(mask_row) - 1) * Int64(key_seq_len) + Int64(col)
            idx = (Int64(row) - 1) * Int64(key_seq_len) + Int64(col)
            m = @inbounds mask[midx]
            v = m != UInt8(1) ? (Float32(@inbounds(src[idx])) * scale) : -10000.0f0
            e = exp(v - maxval)
            sumval += e
        end
        sh[tid] = sumval
        sync_threads()
        offset = blockDim().x ÷ Int32(2)
        while offset > 0
            if tid <= offset
                sh[tid] += sh[tid + offset]
            end
            sync_threads()
            offset ÷= Int32(2)
        end
        sumval = sh[1]

        for col in tid:stride:key_seq_len
            out = 0.0f0
            if full_mask == false
                midx = (Int64(mask_row) - 1) * Int64(key_seq_len) + Int64(col)
                idx = (Int64(row) - 1) * Int64(key_seq_len) + Int64(col)
                m = @inbounds mask[midx]
                v = m != UInt8(1) ? (Float32(@inbounds(src[idx])) * scale) : -10000.0f0
                out = exp(v - maxval) / sumval
            end
            @inbounds dst[(Int64(row) - 1) * Int64(key_seq_len) + Int64(col)] = eltype(dst)(out)
        end
    end
    return
end

