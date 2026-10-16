using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function adjacent_ka!(d_in, d_out, subtract_left::Int32,
                                block_threads::Int32)
    tid = Int32(@index(Local,Linear))-Int32(1)  # 0-based within block
    items_per_thread = Int32(4)
    items_per_block = block_threads * items_per_thread
    block_start = (Int32(@index(Group,Linear))-Int32(1))*items_per_block

    # Load 4 items per thread (consecutive, matching int4 layout: 4 contiguous ints)
    base = block_start + tid * items_per_thread
    @inbounds a0 = d_in[base + Int32(1)]
    @inbounds a1 = d_in[base + Int32(2)]
    @inbounds a2 = d_in[base + Int32(3)]
    @inbounds a3 = d_in[base + Int32(4)]

    # Shared: store thread's last element (subtract_left) or first (subtract_right)
    # to enable cross-thread differences.
    smem = @localmem Int32 (1024,)
    if subtract_left != Int32(0)
        # Thread stores its LAST element for the neighbor to the right to read
        smem[tid + Int32(1)] = a3
    else
        # Thread stores its FIRST element for the neighbor to the left to read
        smem[tid + Int32(1)] = a0
    end
    @synchronize

    if subtract_left != Int32(0)
        # b[i] = a[i] - a[i-1]; a[-1] cross-thread comes from smem[tid-1] (last elem)
        b3 = a3 - a2
        b2 = a2 - a1
        b1 = a1 - a0
        if tid == Int32(0)
            b0 = a0  # first item in block: unchanged
        else
            @inbounds b0 = a0 - smem[tid]  # smem[tid-1+1]
        end
        @inbounds d_out[base + Int32(1)] = b0
        @inbounds d_out[base + Int32(2)] = b1
        @inbounds d_out[base + Int32(3)] = b2
        @inbounds d_out[base + Int32(4)] = b3
    else
        # b[i] = a[i] - a[i+1]; b[last] cross-thread comes from next thread's first elem
        b0 = a0 - a1
        b1 = a1 - a2
        b2 = a2 - a3
        if tid == block_threads - Int32(1)
            b3 = a3
        else
            @inbounds b3 = a3 - smem[tid + Int32(2)]  # next thread's first (a0)
        end
        @inbounds d_out[base + Int32(1)] = b0
        @inbounds d_out[base + Int32(2)] = b1
        @inbounds d_out[base + Int32(3)] = b2
        @inbounds d_out[base + Int32(4)] = b3
    end
end

function factory(input,output)
    kernel=adjacent_ka!(get_backend(input),256)
    return ()->begin
        kernel(input,output,Int32(1),Int32(256);ndrange=1024*256)
        kernel(output,output,Int32(0),Int32(256);ndrange=1024*256)
    end
end
adjacent_study(factory)
