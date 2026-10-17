using CUDA,JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_common.jl"))
function diagonal(index,A,B,path,template)
 tid=Int32((index-1)%32);group=Int32((index-1)÷32);state=JACC.shared(template)
 @inbounds begin
 if tid==0
  combined=Int32((Int64(group)*200000)÷112);state[1]=min(combined,Int32(100000));state[2]=max(combined-Int32(100000),Int32(0));state[3]=state[2];state[4]=state[1];state[5]=0
 end
 JACC.sync_workgroup()
 while state[5]==0
  cx=state[1]-((state[1]-state[3])>>1)-(tid-16);cy=state[2]+((state[4]-state[2])>>1)+(tid-16)
  state[tid+6]=cx>100000||cy<0 ? Int32(0) : cy>=100000||cx<1 ? Int32(1) : Int32(A[cx]<=B[cy+1])
  JACC.sync_workgroup()
  if tid>0&&state[tid+6]!=state[tid+5];state[5]=1;path[group+1]=UInt32(cx);path[group+114]=UInt32(cy);end
  JACC.sync_workgroup()
  if tid==16
   if state[37]!=0;state[3]=cx;state[4]=cy;else;state[1]=cx;state[2]=cy;end
  end
  JACC.sync_workgroup()
 end
 if tid==0&&group==0;path[1]=0;path[114]=0;path[113]=100000;path[226]=100000;end
 end
end
function window(index,A,B,path,C,template)
 tid=Int32((index-1)%128);group=Int32((index-1)÷128);raw=JACC.shared(template)
 @inbounds begin
 T=eltype(A);values=reinterpret(T,raw);m=Int32(1028*(sizeof(T)÷4));negative=T<:AbstractFloat ? floatmin(T) : zero(T);positive=T<:AbstractFloat ? floatmax(T) : typemax(T)
 if tid==0;raw[m+1]=path[group+1];values[1]=negative;end
 if tid==64;raw[m+2]=path[group+114];values[514]=positive;end
 if tid==32;raw[m+3]=path[group+2];values[515]=negative;end
 if tid==96;raw[m+4]=path[group+115];values[1028]=positive;end
 JACC.sync_workgroup()
 while raw[m+1]<raw[m+3]||raw[m+2]<raw[m+4]
  xtop=Int32(raw[m+1]);ytop=Int32(raw[m+2])
  for tile in Int32(0):Int32(3)
   j=tid+1+128tile;values[j+1]=A[xtop+j];values[j+515]=B[ytop+j]
  end
  cstart=xtop+ytop;cstop=Int32(path[group+2]+path[group+115]);JACC.sync_workgroup()
  rank=4tid;left=Int32(0);right=rank;ax=bx=Int32(1)
  while left<=right
   x=left+(right-left)÷2;y=rank-x
   if x>0&&values[x+1]>values[y+516];right=x-1
   elseif y>0&&values[y+515]>=values[x+2];left=x+1
   else;ax=x+1;bx=y+1;break
   end
  end
  ai=values[ax+1];bi=values[bx+515]
  for k in Int32(0):Int32(2)
   if ai>bi;ci=bi;bx+=1;bi=values[bx+515];else;ci=ai;ax+=1;ai=values[ax+1];end
   index=cstart+4tid+k;if index<cstop&&index<200000;C[index+1]=ci;end
  end
  index=cstart+4tid+3;if index<cstop&&index<200000;C[index+1]=min(ai,bi);end
  if tid==127;raw[m+1]+=UInt32(ax-1);raw[m+2]+=UInt32(bx-1);end
  JACC.sync_workgroup()
 end
 end
end
function factory(a,b,path,out)
 words=1028*(sizeof(eltype(a))÷4)+4;state=CUDA.zeros(Int32,37);raw=CUDA.zeros(UInt32,words)
 sd=JACC.launch_spec(;threads=32,blocks=112,shmem_size=148,sync=false);sw=JACC.launch_spec(;threads=128,blocks=112,shmem_size=4words,sync=false)
 ()->begin;JACC.parallel_for(sd,112*32,diagonal,a,b,path,state);JACC.parallel_for(sw,112*128,window,a,b,path,out,raw);CUDA.synchronize();end
end
if abspath(PROGRAM_FILE)==@__FILE__;merge_study(factory);end
