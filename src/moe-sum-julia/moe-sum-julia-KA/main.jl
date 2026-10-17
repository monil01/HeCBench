using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function moe_scalar_ka!(out, input, hidden_size::Int32, topk::Int32)
    token0 = Int32(@index(Group,Linear)) - Int32(1)
    idx0 = Int32(@index(Local,Linear)) - Int32(1)
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

@kernel function moe_vec4_ka!(out, input, hidden_size::Int32, topk::Int32)
    token0 = Int32(@index(Group,Linear)) - Int32(1)
    lane0 = (Int32(@index(Local,Linear)) - Int32(1)) * Int32(4)
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
    kernel=vec ? moe_vec4_ka!(get_backend(input),1024) : moe_scalar_ka!(get_backend(input),1024)
    return ()->kernel(output,input,Int32(1024),Int32(3);ndrange=512*1024)
end
moe_study(factory)
