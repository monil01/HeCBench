using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function lapl!(out,inp,delta,norm,Lx,Ly)
    i=@index(Global,Linear)-1
    if i<Lx*Ly
        x=i%Lx;y=i÷Lx
        @inbounds out[i+1]=norm*inp[i+1]+delta*(inp[y*Lx+((x+1)%Lx)+1]+inp[y*Lx+((Lx+x-1)%Lx)+1]+inp[((y+1)%Ly)*Lx+x+1]+inp[((Ly+y-1)%Ly)*Lx+x+1])
    end
end
function prepare(a,b,Lx,Ly,delta,norm)
    kernel=lapl!(KernelAbstractions.get_backend(a),256)
    (out,inp)->kernel(out,inp,delta,norm,Int32(Lx),Int32(Ly);ndrange=Lx*Ly)
end
exit(study_main(prepare))
