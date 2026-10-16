using CUDA,KernelAbstractions
@kernel function add_ka!(a,b,c,total)
    i=@index(Global,Linear)
    if i<=total
        @inbounds c[i]=a[i]+b[i]
    end
end
@kernel function tv_ka!(a,b,c,m,n,bx)
    block=@index(Group,Linear)-1;tid=@index(Local,Linear)-1
    row0=(block÷bx)*16;col0=(block%bx)*256
    for k in tid:128:4095
        row=row0+k÷256;col=col0+k%256
        if row<m && col<n
            index=row*n+col+1
            @inbounds c[index]=a[index]+b[index]
        end
    end
end
function launch_ka!(::Val{M},a,b,c,m,n) where M
    if M==3
        bx=cld(n,256);by=cld(m,16)
        tv_ka!(get_backend(c),128)(a,b,c,Int32(m),Int32(n),Int32(bx);ndrange=bx*by*128)
    else
        add_ka!(get_backend(c),256)(a,b,c,Int32(m*n);ndrange=m*n)
    end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
vadd_study(launch_ka!)
