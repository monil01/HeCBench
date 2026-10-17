using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
include(joinpath(@__DIR__,"..","postprocess_math.jl"))
@kernel function postprocess_ka!(cls,box,dir,anchors,bottom,bnd,score,count,minx,maxx,miny,maxy,fx,fy,na,nc,nb,threshold,offset)
 loc=Int32(@index(Group,Linear))-Int32(1)
 anchor=Int32(@index(Local,Linear))-Int32(1)
 portable_postprocess!(loc,anchor,cls,box,dir,anchors,bottom,bnd,score,count,minx,maxx,miny,maxy,fx,fy,na,nc,nb,threshold,offset)
end
const POST=postprocess_ka!(CUDABackend(),6)
function launch!(d,threshold)
 POST(d...,ST_RANGES...,ST_FX,ST_FY,ST_NA,ST_NC,ST_NB,threshold,ST_OFFSET;ndrange=ST_K)
end
study_main("ka",launch!)
