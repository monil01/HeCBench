using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
include(joinpath(@__DIR__,"..","postprocess_math.jl"))
const BACKEND=JACC.get_backend(:cuda)
const SPEC=JACC.LaunchSpec{typeof(BACKEND)}(threads=6,blocks=ST_FX*ST_FY,shmem_size=0,sync=false)
@inline function postprocess_jacc!(i,cls,box,dir,anchors,bottom,bnd,score,count,minx,maxx,miny,maxy,fx,fy,na,nc,nb,threshold,offset)
 tid=Int32(i)-Int32(1);loc=tid÷na;anchor=tid%na
 portable_postprocess!(loc,anchor,cls,box,dir,anchors,bottom,bnd,score,count,minx,maxx,miny,maxy,fx,fy,na,nc,nb,threshold,offset)
 return
end
function launch!(d,threshold)
 JACC.parallel_for(postprocess_jacc!,SPEC,ST_K,d...,ST_RANGES...,ST_FX,ST_FY,ST_NA,ST_NC,ST_NB,threshold,ST_OFFSET)
end
study_main("jacc",launch!)
