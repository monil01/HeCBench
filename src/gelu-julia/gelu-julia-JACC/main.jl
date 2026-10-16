using CUDA,JACC

@inline rounded_bias_add(x::Float16,b::Float16)=Float32(Float16(Float32(x)+Float32(b)))
function gelu_f32(x::Float32)
    return 0.5f0 * x * (1.0f0 + tanh(0.79788456f0 * (x + 0.044715f0 * x * x * x)))
end

include(joinpath(@__DIR__,"..","study_common.jl"))
function gelu_base_jacc!(global_x,global_y,src, bias, width::Int32, height::Int32)
    groupx=div(global_x-1,128)+1;groupbatch=global_y
    localid=mod(global_x-1,128)+1
    x = Int32(groupx) - Int32(1)
    batch = Int32(groupbatch) - Int32(1)
    index = batch * width * height + x * width
    y = Int32(localid) - Int32(1)
    while y < width
        v = rounded_bias_add(src[index + y + Int32(1)],bias[y + Int32(1)])
        src[index + y + Int32(1)] = Float16(gelu_f32(v))
        y += Int32(128)
    end
end

function gelu_vec2_jacc!(global_x,global_y,src, bias, width::Int32, height::Int32)
    groupx=div(global_x-1,128)+1;groupbatch=global_y
    localid=mod(global_x-1,128)+1
    x = Int32(groupx) - Int32(1)
    batch = Int32(groupbatch) - Int32(1)
    index = batch * width * height + x * width
    y = (Int32(localid) - Int32(1)) * Int32(2)
    step = Int32(128) * Int32(2)
    while y < width
        v0 = rounded_bias_add(src[index + y + Int32(1)],bias[y + Int32(1)])
        src[index + y + Int32(1)] = Float16(gelu_f32(v0))
        y1 = y + Int32(1)
        if y1 < width
            v1 = rounded_bias_add(src[index + y1 + Int32(1)],bias[y1 + Int32(1)])
            src[index + y1 + Int32(1)] = Float16(gelu_f32(v1))
        end
        y += step
    end
end

function factory(src,bias,mode,width,height,batch)
    spec=JACC.launch_spec(;threads=(128,1),sync=false)
    kernel=mode=="base" ? gelu_base_jacc! : gelu_vec2_jacc!
    return ()->JACC.parallel_for(spec,(height*128,batch),kernel,src,bias,Int32(width),Int32(height))
end
gelu_study(factory)
