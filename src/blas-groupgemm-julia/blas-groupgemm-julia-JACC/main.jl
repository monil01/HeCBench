using CUDA
using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function grouped_kernel!(idx,a,b,c,m,oa,ob,oc,k)
    i=Int32(idx)-Int32(1)
    g=Int32(1)
    while g<Int32(length(m))&&i>=oc[g+Int32(1)]
        g+=Int32(1)
    end
    if i<oc[end]
        row=(i-oc[g])%m[g];col=(i-oc[g])÷m[g];s=0f0
        @inbounds for q in Int32(0):k-Int32(1)
            s+=Float32(a[oa[g]+row+q*m[g]+Int32(1)])*Float32(b[ob[g]+q+col*k+Int32(1)])
        end
        @inbounds c[i+Int32(1)]=Float16(s)
    end
    nothing
end
function study_prepare(a,b,c,m,oa,ob,oc,k,n)
    dm,doa,dob,doc=CuArray(m),CuArray(oa),CuArray(ob),CuArray(oc);backend=JACC.get_backend(:cuda);spec=JACC.LaunchSpec{typeof(backend)}(;threads=256,blocks=cld(length(c),256),shmem_size=0,sync=false)
    ()->JACC.parallel_for(grouped_kernel!,spec,length(c),a,b,c,dm,doa,dob,doc,Int32(k))
end
exit(study_main(study_prepare))
