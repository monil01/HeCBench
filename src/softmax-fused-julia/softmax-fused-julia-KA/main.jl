using CUDA,KernelAbstractions
@kernel unsafe_indices=true function portable_softmax!(dst, src, mask, scale::Float32,
                                       rows::Int64, key_seq_len::Int32,
                                       query_seq_len::Int32, pad_batches::Int32, heads::Int32)
    row = Int64(@index(Group,Linear))
    tid = Int32(@index(Local,Linear))
    stride = Int32(256)
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
        sh = @localmem(Float32,(256,))
        sh[tid] = maxval
        @synchronize()
        offset = Int32(128)
        while offset > 0
            if tid <= offset
                sh[tid] = max(sh[tid], sh[tid + offset])
            end
            @synchronize()
            offset ÷= Int32(2)
        end
        maxval = sh[1]
        @synchronize() # all warps consume max before shared buffer reuse
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
        @synchronize()
        offset = Int32(128)
        while offset > 0
            if tid <= offset
                sh[tid] += sh[tid + offset]
            end
            @synchronize()
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
end

include(joinpath(@__DIR__,"..","study_common.jl"))
const PORTABLE=portable_softmax!(CUDABackend(),256)
function launch!(f)
 b,h,q,c,pad=f[6:10];rows=Int64(b)*Int64(h)*Int64(q)
 PORTABLE(f[3],f[1],f[2],f[5],rows,c,q,pad,h;ndrange=rows*256)
end
study_main("ka",launch!)
