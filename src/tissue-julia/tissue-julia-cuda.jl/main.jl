using CUDA
# Accepted namedstep1 usesonewriter/point; step4CLI peerupdatehazard isnotcertified.
function tissue_kernel!(points,g,b,output,prev,qt,nnt,capacity,step,species)
    i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)-Int32(1)
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
    ()->(@cuda threads=256 blocks=cld(nnt,256) tissue_kernel!(points,g,b,output,prev,qt,Int32(nnt),Int32(capacity),Int32(1),Int32(species)))
end
if abspath(PROGRAM_FILE)==@__FILE__
    include(joinpath(@__DIR__,"..","study_common.jl"));exit(study_main(study_prepare))
end
