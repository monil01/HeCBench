using CUDA
import JACC
JACC.@init_backend
function add_jacc!(i,a,b,c)
    @inbounds c[i]=a[i]+b[i]
    return
end
function tv_jacc!(i,a,b,c,m,n,bx)
    block=(i-1)÷128;tid=(i-1)%128
    row0=(block÷bx)*16;col0=(block%bx)*256
    for k in tid:128:4095
        row=row0+k÷256;col=col0+k%256
        if row<m && col<n
            index=row*n+col+1
            @inbounds c[index]=a[index]+b[index]
        end
    end
    return
end
const VADD_SPEC=JACC.launch_spec(threads=256,sync=false)
const TV_SPEC=JACC.launch_spec(threads=128,sync=false)
function launch_jacc!(::Val{M},a,b,c,m,n) where M
    if M==3
        bx=cld(n,256);by=cld(m,16)
        JACC.parallel_for(TV_SPEC,bx*by*128,tv_jacc!,a,b,c,Int32(m),Int32(n),Int32(bx))
    else
        JACC.parallel_for(VADD_SPEC,m*n,add_jacc!,a,b,c)
    end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
vadd_study(launch_jacc!)
