using CUDA,JACC
function portable_softmax!(i,dst, src, mask, scale::Float32,
                                       rows::Int64, key_seq_len::Int32,
                                       query_seq_len::Int32, pad_batches::Int32, heads::Int32, template)
    gid=Int64(i)-1
    row = gid÷256+1
    tid = Int32(gid%256+1)
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
        sh = JACC.shared(template)
        sh[tid] = maxval
        JACC.sync_workgroup()
        offset = Int32(128)
        while offset > 0
            if tid <= offset
                sh[tid] = max(sh[tid], sh[tid + offset])
            end
            JACC.sync_workgroup()
            offset ÷= Int32(2)
        end
        maxval = sh[1]
        JACC.sync_workgroup() # all warps consume max before shared buffer reuse
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
        JACC.sync_workgroup()
        offset = Int32(128)
        while offset > 0
            if tid <= offset
                sh[tid] += sh[tid + offset]
            end
            JACC.sync_workgroup()
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

include(joinpath(@__DIR__,"..","study_common.jl"))
const BACKEND=JACC.get_backend(:cuda)
const TEMPLATE=CUDA.zeros(Float32,256)
const SPECS=Dict{Int64,Any}()
function launch!(f)
 b,h,q,c,pad=f[6:10];rows=Int64(b)*Int64(h)*Int64(q)
 spec=get!(SPECS,rows) do
  JACC.LaunchSpec{typeof(BACKEND)}(;threads=256,blocks=Int(rows),shmem_size=1024,sync=false)
 end
 JACC.parallel_for(portable_softmax!,spec,Int(rows)*256,f[3],f[1],f[2],f[5],rows,c,q,pad,h,TEMPLATE)
end
study_main("jacc",launch!)
