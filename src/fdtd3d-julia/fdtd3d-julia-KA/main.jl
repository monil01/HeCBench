using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function fdtd!(out,inp,coeff,nx,ny,nz,r)
    x,y,z=@index(Global,NTuple)
    if x<=nx && y<=ny && z<=nz
        ox=nx+2r;oy=ny+2r;sz=ox*oy
        i=x+r+(y+r-1)*ox+(z+r-1)*sz
        @inbounds begin
            value=coeff[1]*inp[i]
            for k in 1:r
                c=coeff[k+1]
                value+=c*(inp[i+k]+inp[i-k])
                value+=c*(inp[i+k*ox]+inp[i-k*ox])
                value+=c*(inp[i+k*sz]+inp[i-k*sz])
            end
            out[i]=value
        end
    end
end
function prepare(a,b,c,nx,ny,nz,r)
    kernel=fdtd!(KernelAbstractions.get_backend(a),(8,8,4))
    (out,inp)->kernel(out,inp,c,Int32(nx),Int32(ny),Int32(nz),Int32(r);ndrange=(nx,ny,nz))
end
exit(study_main(prepare))
