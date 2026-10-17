using CUDA
using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function tissue_kernel!(idx,points,g,b,output,prev,qt,nnt,capacity,step,species)
    i=Int32(idx)-Int32(1)
    if i<step*nnt
        point=i÷step;part=i%step;p=0f0;nnt2=Int32(2)*nnt
        if point<nnt
            @inbounds begin
                ix=points[point+Int32(1)];iy=points[point+nnt+Int32(1)];iz=points[point+nnt2+Int32(1)]
                for j in part:step:nnt-Int32(1)
                    jx=points[j+Int32(1)];jy=points[j+nnt+Int32(1)];jz=points[j+nnt2+Int32(1)];index=abs(jx-ix)+abs(jy-iy)+abs(jz-iz)+(species-Int32(1))*capacity+Int32(1)
                    p+=g[index]*prev[j+Int32(1)]+b[index]*qt[j+Int32(1)]
                end
                if part==Int32(0);output[point+Int32(1)]=p;end
            end
        end
        for stage in Int32(1):step-Int32(1)
            if part==stage&&point<nnt
                @inbounds output[point+Int32(1)]+=p
            end
        end
    end
    nothing
end
function study_prepare(points,g,b,output,prev,qt,nnt,capacity,species)
    backend=JACC.get_backend(:cuda);spec=JACC.LaunchSpec{typeof(backend)}(;threads=256,blocks=cld(nnt,256),shmem_size=0,sync=false)
    ()->JACC.parallel_for(tissue_kernel!,spec,nnt,points,g,b,output,prev,qt,Int32(nnt),Int32(capacity),Int32(1),Int32(species))
end
exit(study_main(study_prepare))
