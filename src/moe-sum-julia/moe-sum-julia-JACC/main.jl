using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function moe_scalar_jacc!(index,out, input, hidden_size::Int32, topk::Int32)
    token0 = Int32(div(index-1,1024)+1) - Int32(1)
    idx0 = Int32(mod(index-1,1024)+1) - Int32(1)
    output_base = token0 * hidden_size
    input_base = output_base * topk

    while idx0 < hidden_size
        x = 0.0f0
        for k in Int32(0):topk-Int32(1)
            @inbounds x += input[input_base + k * hidden_size + idx0 + Int32(1)]
        end
        @inbounds out[output_base + idx0 + Int32(1)] = x
        idx0 += Int32(1024)
    end
end

function moe_vec4_jacc!(index,out, input, hidden_size::Int32, topk::Int32)
    token0 = Int32(div(index-1,1024)+1) - Int32(1)
    lane0 = (Int32(mod(index-1,1024)+1) - Int32(1)) * Int32(4)
    stride = Int32(1024) * Int32(4)
    output_base = token0 * hidden_size
    input_base = output_base * topk

    while lane0 < hidden_size
        for c in Int32(0):Int32(3)
            idx0 = lane0 + c
            if idx0 < hidden_size
                x = 0.0f0
                for k in Int32(0):topk-Int32(1)
                    @inbounds x += input[input_base + k * hidden_size + idx0 + Int32(1)]
                end
                @inbounds out[output_base + idx0 + Int32(1)] = x
            end
        end
        lane0 += stride
    end
end

function factory(input,output,vec)
    spec=JACC.launch_spec(;threads=1024,shmem_size=0,sync=false)
    kernel=vec ? moe_vec4_jacc! : moe_scalar_jacc!
    return ()->JACC.parallel_for(spec,512*1024,kernel,output,input,Int32(1024),Int32(3))
end
moe_study(factory)
