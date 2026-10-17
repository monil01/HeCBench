using CUDA,JACC
const TPB=Int32(256)
const COLS_PER_BLK=Int32(32)
include(joinpath(@__DIR__,"..","study_common.jl"))
function sop_jacc!(gx,gy,template,std, data, D::Int32, N::Int32)
    rows_per_blk_per_iter = TPB ÷ COLS_PER_BLK
    tx0 = Int32(mod(gx-1,256))
    this_col = tx0 % COLS_PER_BLK
    this_row = tx0 ÷ COLS_PER_BLK
    col = this_col + (Int32(gy) - Int32(1)) * COLS_PER_BLK
    row = this_row + (Int32(div(gx-1,256)+1) - Int32(1)) * rows_per_blk_per_iter
    stride = rows_per_blk_per_iter * Int32(128)

    thread_sum = 0.0f0
    r = row
    while r < N
        if col < D
            v = data[Int64(r) * Int64(D) + Int64(col) + 1]
            thread_sum += v * v
        end
        r += stride
    end

    sstd=JACC.shared(template)
    if tx0 < COLS_PER_BLK
        sstd[tx0 + 1] = 0.0f0
    end
    JACC.sync_workgroup()

    JACC.@atomic sstd[this_col + 1] += thread_sum
    JACC.sync_workgroup()

    if tx0 < COLS_PER_BLK && col < D
        JACC.@atomic std[col + 1] += sstd[tx0 + 1]
    end
    return
end

function sample_jacc!(index,std, D::Int32, sample_size::Int32)
    i = Int32(index)-Int32(1)
    if i < D
        std[i + 1] = sqrt(std[i + 1] / Float32(sample_size))
    end
    return
end

function factory(output,input)
    template=CUDA.zeros(Float32,32)
    spec2=JACC.launch_spec(;threads=(256,1),shmem_size=128,sync=false)
    spec1=JACC.launch_spec(;threads=256,shmem_size=0,sync=false)
    return ()->begin
        fill!(output,0f0)
        JACC.parallel_for(spec2,(128*256,8),sop_jacc!,template,output,input,Int32(256),Int32(4096))
        JACC.parallel_for(spec1,256,sample_jacc!,output,Int32(256),Int32(4095))
    end
end
stddev_study(factory)
