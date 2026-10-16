using CUDA
using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function morphology_kernel!(g,dst,src,W::Int32,H::Int32,hs::Int32,vs::Int32,op::Int32,axis::Int32)
    i=Int32(g)-Int32(1)
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
    backend=JACC.get_backend(:cuda)
    spec=JACC.LaunchSpec{typeof(backend)}(;threads=256,blocks=cld(STUDY_W*STUDY_H,256),shmem_size=0,sync=false)
    function cycle!(img,tmp)
        for op in (Int32(1),Int32(2))
            CUDA.fill!(tmp,UInt8(0))
            JACC.parallel_for(morphology_kernel!,spec,STUDY_W*STUDY_H,tmp,img,Int32(STUDY_W),Int32(STUDY_H),Int32(STUDY_SEL),Int32(STUDY_SEL),op,Int32(1))
            JACC.parallel_for(morphology_kernel!,spec,STUDY_W*STUDY_H,img,tmp,Int32(STUDY_W),Int32(STUDY_H),Int32(STUDY_SEL),Int32(STUDY_SEL),op,Int32(2))
        end
        nothing
    end
    cycle!
end
exit(study_main(prepare))
