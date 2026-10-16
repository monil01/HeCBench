using CUDA,KernelAbstractions,Atomix
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function accuracy_ka!(N::Int32, D::Int32, top_k::Int32,
                          Xdata, labelData, accuracy_out)
    tid = Int32(@index(Local,Linear))
    bdim = Int32(256)

    smem = @localmem Int32 (256,)

    count = Int32(0)

    row = Int32(@index(Group,Linear))-Int32(1)
    grid = Int32(1024)
    while row < N
        label = labelData[row + Int32(1)]
        label_pred = Xdata[row * D + label + Int32(1)]
        ngt = Int32(0)
        col = tid - Int32(1)
        @inbounds while col < D
            pred = Xdata[row * D + col + Int32(1)]
            if (pred > label_pred) || (pred == label_pred && col <= label)
                ngt += Int32(1)
            end
            col += bdim
        end

        # block reduce ngt via shared memory
        smem[tid] = ngt
        @synchronize
        stride = bdim >> 1
        while stride > Int32(0)
            if tid <= stride
                smem[tid] += smem[tid + stride]
            end
            @synchronize
            stride >>= 1
        end

        if tid == Int32(1)
            if smem[1] <= top_k
                count += Int32(1)
            end
        end
        @synchronize

        row += grid
    end

    if tid == Int32(1)
        Atomix.@atomic accuracy_out[1] += count
    end
end

function factory(input,labels,output)
    kernel=accuracy_ka!(get_backend(input),256)
    return ()->kernel(Int32(4096),Int32(128),Int32(5),input,labels,output;ndrange=1024*256)
end
accuracy_study(factory)
