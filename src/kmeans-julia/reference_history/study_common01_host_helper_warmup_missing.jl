using CUDA,Libdl
function nearest(x,i,f,cent,k)
 index=0;best=typemax(Float32)
 @inbounds for j in 0:k-1
  dist=0f0
  for a in 1:f;d=x[(i-1)*f+a]-cent[j*f+a];dist+=d*d;end
  if dist<best;best=dist;index=j;end
 end
 index
end
function update!(c,cent,mem,prev)
 k=div(length(cent),c.f);s=zeros(Float32,length(cent));counts=zeros(Int32,k);delta=0
 @inbounds for i in 1:c.n
  j=Int(mem[i])+1;1<=j<=k||error("memberrange");counts[j]+=1
  if mem[i]!=prev[i];delta+=1;prev[i]=mem[i];end
  for a in 1:c.f;s[(j-1)*c.f+a]+=c.host[(i-1)*c.f+a];end
 end
 @inbounds for j in 1:k
  if counts[j]>0
   for a in 1:c.f;cent[(j-1)*c.f+a]=s[(j-1)*c.f+a]/Float32(counts[j]);end
  end
 end
 delta
end
function rmse(c,cent)
 s=0f0;k=div(length(cent),c.f)
 @inbounds for i in 1:c.n
  j=nearest(c.host,i,c.f,cent,k);dist=0f0
  for a in 1:c.f;d=c.host[(i-1)*c.f+a]-cent[j*c.f+a];dist+=d*d;end
  s+=dist
 end
 sqrt(s/Float32(c.n))
end
function study(alloc,first,second)
 cs=[]
 for n in (17,257,4099)
  f=n==17 ? 3 : n==257 ? 4 : 7;lo=n==17 ? 2 : 5;hi=n==17 ? 4 : 15;restarts=n==17 ? 2 : 10
  host=Float32[2f0*Float32(mod(i,5))+Float32((a+1)*(mod(i,5)+1))/32f0+(n==17 ? 0f0 : Float32(mod(i,7)-3)/256f0) for i in 0:n-1 for a in 0:f-1]
  rx=alloc(fill(-777f0,n*f+128));rs=alloc(fill(-777f0,n*f+128));rc=alloc(fill(-777f0,hi*f+128));rm=alloc(fill(Int32(-777),n+128))
  push!(cs,(n=n,f=f,lo=lo,hi=hi,restarts=restarts,host=host,rx=rx,rs=rs,rc=rc,rm=rm,x=view(rx,65:64+n*f),sw=view(rs,65:64+n*f),cent=view(rc,65:64+hi*f),mem=view(rm,65:64+n)))
 end
 function reset(c)
  fill!(c.rx,-777f0);copyto!(c.x,c.host);fill!(c.rs,-777f0);fill!(c.rc,-777f0);fill!(c.rm,Int32(-777))
 end
 for c in cs;reset(c);first(c);copyto!(view(c.cent,1:c.lo*c.f),c.host[1:c.lo*c.f]);second(c,c.lo);CUDA.device_synchronize();end
 check=Libdl.dlsym(Libdl.dlopen(joinpath(pwd(),"studycheck.so")),:study_check)
 for trial in 1:5
  out=Float64[3];seconds=0.;calls=0
  for c in cs
   reset(c);append!(out,Float64[c.n,c.f,c.lo,c.hi,c.restarts]);append!(out,Array(c.x))
  end
  for c in cs
   bestk=0;bestloop=0;bestrm=typemax(Float32)
   for k in c.lo:c.hi
    push!(out,k);CUDA.device_synchronize();start=time_ns();first(c);CUDA.device_synchronize();seconds+=(time_ns()-start)*1e-9;calls+=1
    append!(out,Array(c.sw));initial=collect(0:c.n-1);left=c.n
    for restart in 0:c.restarts-1
     push!(out,restart);cent=Vector{Float32}(undef,k*c.f)
     for j in 0:k-1
      left>0||error("initialunderflow")
      for a in 1:c.f;cent[j*c.f+a]=c.host[initial[j+1]*c.f+a];end
      initial[j+1],initial[left]=initial[left],initial[j+1];left-=1
     end
     prev=fill(Int32(-1),c.n);loop=0;iter=0
     while true
      push!(out,iter);start=time_ns()
      copyto!(view(c.cent,1:length(cent)),cent);second(c,k);mem=Array(c.mem);delta=update!(c,cent,mem,prev);CUDA.device_synchronize();seconds+=(time_ns()-start)*1e-9;calls+=1
      append!(out,Array(view(c.cent,1:length(cent))));append!(out,mem);push!(out,delta);append!(out,cent);iter+=1
      if !(delta>.001f0&&loop<500);break;end
      loop+=1
     end
     push!(out,iter);start=time_ns();rms=rmse(c,cent);truncated=trunc(Int,rms)
     if truncated<bestrm;bestrm=Float32(truncated);bestk=k;bestloop=restart;end
     seconds+=(time_ns()-start)*1e-9;push!(out,rms);push!(out,truncated);append!(out,cent)
    end
   end
   append!(out,Float64[bestk,bestloop,bestrm]);reinterpret(UInt32,Array(c.x))==reinterpret(UInt32,c.host)||error("readonlyinput")
   for raw in (c.rx,c.rs,c.rc,c.rm);h=Array(raw);append!(out,h[1:64]);append!(out,h[end-63:end]);end
  end
  path="trial$(trial).bin";open(io->write(io,out),path,"w");met=zeros(Float64,3);ccall(check,Cint,(Cstring,Ptr{Float64}),path,met)==0||error("independent/fullpipelinecheck")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":true,\"checked_values\":",length(out),",\"max_abs_error\":",met[1],",\"max_rel_error\":",met[2],",\"independent_centroid_max_abs\":",met[3],",\"timing_region\":\"sum_preallocated_transpose_iteration_upload_kernel_download_hostupdate_RMSE\",\"jit_excluded\":true,\"internal_repeats\":",calls,",\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":",length(out),",\"byte_order\":\"little\"}}")
 end
end
