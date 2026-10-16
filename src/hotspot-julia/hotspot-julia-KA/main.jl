using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function hotspot!(out,cur,power,dc,rx,ry,rz,L)
    x,y=@index(Global,NTuple)
    if x<=L && y<=L
        n=max(y-1,1);s=min(y+1,L);w=max(x-1,1);e=min(x+1,L);i=(y-1)*L+x
        @inbounds begin
            t=cur[i]
            out[i]=t+dc*(power[i]+(cur[(s-1)*L+x]+cur[(n-1)*L+x]-2f0*t)*ry+(cur[(y-1)*L+e]+cur[(y-1)*L+w]-2f0*t)*rx+(80f0-t)*rz)
        end
    end
end
function prepare(a,b,p,dc,rx,ry,rz)
    kernel=hotspot!(KernelAbstractions.get_backend(a),(16,16))
    (out,inp)->kernel(out,inp,p,dc,rx,ry,rz,Int32(STUDY_L);ndrange=(STUDY_L,STUDY_L))
end
exit(study_main(prepare))
