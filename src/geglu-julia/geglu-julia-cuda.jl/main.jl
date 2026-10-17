using CUDA
using SpecialFunctions
function geglu_kernel!(input,output,width)
    tid=Int32(threadIdx().x)-Int32(1);block=Int32(blockIdx().x)-Int32(1)
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
    blocks=rows*width÷2560
    ()->(@cuda threads=160 blocks=blocks geglu_kernel!(input,output,Int32(width)))
end
if abspath(PROGRAM_FILE)==@__FILE__
    include(joinpath(@__DIR__,"..","study_common.jl"));exit(study_main(study_prepare))
end
