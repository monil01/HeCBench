using CUDA,KernelAbstractions,Atomix
const TPB=Int32(256)
const COLS_PER_BLK=Int32(32)
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function sop_ka!(std, data, D::Int32, N::Int32)
    rows_per_blk_per_iter = TPB ÷ COLS_PER_BLK
    tx0 = Int32(@index(Local,Linear))-Int32(1)
    this_col = tx0 % COLS_PER_BLK
    this_row = tx0 ÷ COLS_PER_BLK
    col = this_col + (Int32(@index(Group,NTuple)[2]) - Int32(1)) * COLS_PER_BLK
    row = this_row + (Int32(@index(Group,NTuple)[1]) - Int32(1)) * rows_per_blk_per_iter
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

    sstd = @localmem Float32 (32,)
    if tx0 < COLS_PER_BLK
        sstd[tx0 + 1] = 0.0f0
    end
    @synchronize

    Atomix.@atomic sstd[this_col + 1] += thread_sum
    @synchronize

    if tx0 < COLS_PER_BLK && col < D
        Atomix.@atomic std[col + 1] += sstd[tx0 + 1]
    end
end

@kernel function sample_ka!(std, D::Int32, sample_size::Int32)
    i = Int32(@index(Global,Linear))-Int32(1)
    if i < D
        std[i + 1] = sqrt(std[i + 1] / Float32(sample_size))
    end
end

function factory(output,input)
    sop=sop_ka!(get_backend(input),(256,1));sample=sample_ka!(get_backend(input),256)
    return ()->begin
        fill!(output,0f0)
        sop(output,input,Int32(256),Int32(4096);ndrange=(128*256,8))
        sample(output,Int32(256),Int32(4095);ndrange=256)
    end
end
stddev_study(factory)
