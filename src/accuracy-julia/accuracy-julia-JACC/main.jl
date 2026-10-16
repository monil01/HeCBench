using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function accuracy_jacc!(index,template,N::Int32, D::Int32, top_k::Int32,
                          Xdata, labelData, accuracy_out)
    tid = Int32(mod(index-1,256)+1)
    bdim = Int32(256)

    smem = JACC.shared(template)

    count = Int32(0)

    row = Int32(div(index-1,256))
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
        JACC.sync_workgroup()
        stride = bdim >> 1
        while stride > Int32(0)
            if tid <= stride
                smem[tid] += smem[tid + stride]
            end
            JACC.sync_workgroup()
            stride >>= 1
        end

        if tid == Int32(1)
            if smem[1] <= top_k
                count += Int32(1)
            end
        end
        JACC.sync_workgroup()

        row += grid
    end

    if tid == Int32(1)
        JACC.@atomic accuracy_out[1] += count
    end
end

function factory(input,labels,output)
    template=CUDA.zeros(Int32,256)
    spec=JACC.launch_spec(;threads=256,shmem_size=1024,sync=false)
    return ()->JACC.parallel_for(spec,1024*256,accuracy_jacc!,template,Int32(4096),Int32(128),Int32(5),input,labels,output)
end
accuracy_study(factory)
