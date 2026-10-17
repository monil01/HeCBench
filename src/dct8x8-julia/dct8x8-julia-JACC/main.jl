using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
include(joinpath(@__DIR__,"..","dct8_math.jl"))
const STUDY_TEMPLATE=CUDA.zeros(Float32,528)
@inline function study_tile!(gi,gj,dst,src,s,h,w,dir,template)
 l=((gi-1)%32+1,(gj-1)%2+1);g=((gi-1)÷32+1,(gj-1)÷2+1)
 tile=JACC.shared(template)
 lx=l[1]-1;ly=8*(l[2]-1);mx=lx%8;gx=(g[1]-1)*32+lx;gy=(g[2]-1)*16+ly
 lv=ly*33+lx+1;lh=(ly+mx)*33+lx-mx+1;valid=gx-mx+7<w&&gy+7<h
 if valid
  for i in 0:7;@inbounds tile[lv+i*33]=src[(gy+i)*s+gx+1];end
 end
 JACC.sync_workgroup()
 if valid
  d=(tile[lh+0],tile[lh+1],tile[lh+2],tile[lh+3],tile[lh+4],tile[lh+5],tile[lh+6],tile[lh+7]);z=study_butterfly(d,dir)
  for i in 0:7;@inbounds tile[lh+i]=z[i+1];end
 end
 JACC.sync_workgroup()
 if valid
  d=(tile[lv+0],tile[lv+33],tile[lv+66],tile[lv+99],tile[lv+132],tile[lv+165],tile[lv+198],tile[lv+231]);z=study_butterfly(d,dir)
  for i in 0:7;@inbounds dst[(gy+i)*s+gx+1]=z[i+1];end
 end
end
function study_launch!(y,x,s,h,w,dir)
 backend=JACC.get_backend(:cuda)
 spec=JACC.LaunchSpec{typeof(backend)}(;threads=(32,2),blocks=(cld(w,32),cld(h,16)),shmem_size=2112,sync=false)
 JACC.parallel_for(study_tile!,spec,(cld(w,32)*32,cld(h,16)*2),y,x,s,h,w,dir,STUDY_TEMPLATE)
end
study_main(study_launch!)
