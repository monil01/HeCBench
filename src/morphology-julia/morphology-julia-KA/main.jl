using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function morphology_kernel!(dst,src,W::Int32,H::Int32,hs::Int32,vs::Int32,op::Int32,axis::Int32)
    i=Int32(@index(Global,Linear))-Int32(1)
    if i<W*H
        x=i%W;y=i÷W;sel=axis==Int32(1) ? hs : vs;half=sel÷Int32(2)
        position=axis==Int32(1) ? x : y;limit=axis==Int32(1) ? W : H
        if position<half || position>=limit-half
            @inbounds dst[i+Int32(1)]=axis==Int32(1) ? UInt8(0) : (op==Int32(1) ? UInt8(255) : UInt8(0))
        else
            acc=op==Int32(1) ? UInt8(0) : UInt8(255)
            for k in -half:half
                j=axis==Int32(1) ? i+k : i+k*W
                @inbounds value=src[j+Int32(1)]
                acc=op==Int32(1) ? max(acc,value) : min(acc,value)
            end
            @inbounds dst[i+Int32(1)]=acc
        end
    end
    nothing
end
function prepare(a,b)
    kernel=morphology_kernel!(KernelAbstractions.get_backend(a),256)
    function cycle!(img,tmp)
        for op in (Int32(1),Int32(2))
            CUDA.fill!(tmp,UInt8(0))
            kernel(tmp,img,Int32(STUDY_W),Int32(STUDY_H),Int32(STUDY_SEL),Int32(STUDY_SEL),op,Int32(1);ndrange=STUDY_W*STUDY_H)
            kernel(img,tmp,Int32(STUDY_W),Int32(STUDY_H),Int32(STUDY_SEL),Int32(STUDY_SEL),op,Int32(2);ndrange=STUDY_W*STUDY_H)
        end
        nothing
    end
    cycle!
end
exit(study_main(prepare))
