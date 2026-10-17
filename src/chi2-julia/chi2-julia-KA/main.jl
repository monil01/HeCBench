using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function chi_kernel!(input,output,cols,ncases,ncontrols)
    col=Int32(@index(Global,Linear))-Int32(1)
    if col<cols
        c0=Int32(1);c1=Int32(1);c2=Int32(1);n0=Int32(1);n1=Int32(1);n2=Int32(1)
        @inbounds for row in Int32(0):ncases-Int32(1)
            y=input[row*cols+col+Int32(1)]
            c0+=Int32(y==UInt8(48));c1+=Int32(y==UInt8(49));c2+=Int32(y==UInt8(50))
        end
        @inbounds for row in ncases:ncases+ncontrols-Int32(1)
            y=input[row*cols+col+Int32(1)]
            n0+=Int32(y==UInt8(48));n1+=Int32(y==UInt8(49));n2+=Int32(y==UInt8(50))
        end
        ct=c0+c1+c2;nt=n0+n1+n2;total=ct+nt;s=0f0
        for p in Int32(0):Int32(2)
            ca=p==Int32(0) ? c0 : (p==Int32(1) ? c1 : c2);co=p==Int32(0) ? n0 : (p==Int32(1) ? n1 : n2);count=Float32(ca)+Float32(co)
            ce=Float32(ct)*count/Float32(total);ne=Float32(nt)*count/Float32(total);d1=Float32(ca)-ce;d2=Float32(co)-ne;s+=d1*d1/ce+d2*d2/ne
        end
        @inbounds output[col+Int32(1)]=s
    end
    nothing
end
function study_prepare(input,output,rows,cols,ncases,ncontrols)
    k=chi_kernel!(CUDABackend(),256)
    ()->k(input,output,Int32(cols),Int32(ncases),Int32(ncontrols);ndrange=cols)
end
exit(study_main(study_prepare))
