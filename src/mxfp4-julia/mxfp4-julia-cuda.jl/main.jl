using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function scaled_fp4_kernel!(input,output,n::Int32)
    i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
    if i<=n
        value=@inbounds input[i];maxval=abs(Float32(value))
        for mask in (1,2,4,8,16)
            maxval=max(maxval,CUDA.shfl_xor_sync(UInt32(0xffffffff),maxval,Int32(mask)))
        end
        @inbounds output[i]=scaled_fp4(value,Float16(maxval))
    end
    return
end
factory(input,output,n)=()->(@cuda threads=128 blocks=div(n,128) scaled_fp4_kernel!(input,output,Int32(n)))
function main(args)
    length(args)==2||error("Usage: main.jl <elements multiple128> <repeat>")
    println("FP16 scaled FP4 simulation; BF16 is not implemented/verified by this reference")
    mxfp4_study(factory;n=parse(Int,args[1]),repeats=parse(Int,args[2]))
end
if abspath(PROGRAM_FILE)==@__FILE__;main(ARGS);end
