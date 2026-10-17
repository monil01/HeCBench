using CUDA
using KernelAbstractions
using SpecialFunctions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function geglu_kernel!(input,output,width)
    tid=Int32(@index(Local,Linear))-Int32(1);block=Int32(@index(Group,Linear))-Int32(1)
    chunks=width÷Int32(1280)
    for iteration in Int32(0):Int32(1)
        chunk=block*Int32(2)+iteration;row=chunk÷chunks;start=((chunk%chunks)*Int32(160)+tid)*Int32(8)
        @inbounds for lane in Int32(0):Int32(7)
            x=input[row*Int32(2)*width+start+lane+Int32(1)];gate=input[row*Int32(2)*width+width+start+lane+Int32(1)]
            value=gate*.5f0*(1f0+erf(gate*0.7071067811865476f0));output[row*width+start+lane+Int32(1)]=x*value
        end
    end
    nothing
end
function study_prepare(input,output,rows,width)
    blocks=rows*width÷2560;k=geglu_kernel!(CUDABackend(),160)
    ()->k(input,output,Int32(width);ndrange=blocks*160)
end
println("STUDY_PACKAGE SpecialFunctions ",pkgversion(SpecialFunctions))
exit(study_main(study_prepare))
