using Libdl,Printf
function readcases()
 open(joinpath(@__DIR__,"fixtures.bin")) do f
  cs=[]
  for _ in 1:read(f,UInt32)
   v=read!(f,Vector{UInt32}(undef,10));r=read!(f,Vector{Float32}(undef,5))
   C=read!(f,Vector{UInt32}(undef,v[9]));ex=read!(f,Vector{Float64}(undef,v[10]))
   push!(cs,(h=Int32(v[1]),w=Int32(v[2]),dim=Int32(v[3]),iterations=Int(v[4]),weight=Float32(v[5]),seed=v[6],runs=Int(v[7]),reduce_step=Int(v[8]),temp=r[1],chance=r[2],reduce_factor=r[3],density=r[4],threshold=r[5],C=C,expected=ex))
  end
  cs
 end
end
function padded(c)
 pw=cld(c.w,32)*32;v=zeros(UInt32,cld(c.h,32)*pw)
 for i in 0:cld(c.h,32)-1,j in 0:c.w-1;v[i*pw+j+1]=c.C[i*c.w+j+1];end
 v
end
function guarded(::Type{T},n) where T
 parent=CuArray(fill(reinterpret(T,0xdeadbeef),n+16));(parent,view(parent,9:n+8))
end
function runstudy(cs)
 prefix=Float64[];out=Float64[];total=0.
 for c in cs
  pw=Int32(cld(c.w,32)*32);pad=padded(c)
  pa,A=guarded(UInt32,c.h);pb,B=guarded(UInt32,c.w);pc,C=guarded(UInt32,length(pad));pe,E=guarded(Float32,1)
  copyto!(C,pad);append!(prefix,Float64.(Array(C)))
  bestA=Vector{UInt32}(undef,c.h);bestB=Vector{UInt32}(undef,c.w);CUDA.pin(bestA);CUDA.pin(bestB)
  distance=Float32[0];CUDA.pin(distance);scores=Float32[];best=typemax(Float32)
  state=rnginit(c.seed)
  CUDA.device_synchronize();t=time_ns()
  for run in 1:c.runs
   seed,state=rngnext(state);runseed,state=rngnext(state)
   println("Starting run ",run-1," in slot 0 with seed ",runseed)
   threshold=sqrt(1f0-(1f0-c.density)^(1f0/Float32(c.dim)))
   launchinit(A,c.h,c.dim,seed,threshold);launchinit(B,c.w,c.dim,seed+UInt32(c.h),threshold)
   fill!(E,0f0);launchdist(A,B,C,c.h,c.w,pw,c.weight,E)
   GC.@preserve distance begin;Base.unsafe_copyto!(pointer(distance),pointer(E),1;async=true);end
   CUDA.synchronize();polled=distance[1];previous=polled;stuck=0;temp=c.temp;weight=c.weight;s=rnginit(runseed)
   for it in 1:c.iterations
    (polled> -1f0 && temp> -1f0 && stuck<10000)||break
    r,s=rngnext(s);start=Int32(r%UInt32(c.h))÷16*16;r,s=rngnext(s)
    launchupdate(false,A,B,C,c.h,c.w,pw,c.dim,start,E,r+UInt32(it),temp/10f0,c.chance,Int32(2),weight)
    r,s=rngnext(s);start=Int32(r%UInt32(c.w))÷16*16;r,s=rngnext(s)
    launchupdate(true,A,B,C,c.h,c.w,pw,c.dim,start,E,r+UInt32(it),temp/10f0,c.chance,Int32(2),weight)
    CUDA.synchronize()
    if it%100==0
     GC.@preserve distance begin;Base.unsafe_copyto!(pointer(distance),pointer(E),1;async=true);end
     CUDA.synchronize();polled=distance[1];stuck=polled==previous ? stuck+100 : 0;previous=polled
    end
    if it%c.reduce_step==0
     temp*=c.reduce_factor
     if weight>1f0;weight*=c.reduce_factor;end
     weight=max(weight,1f0)
    end
   end
   fill!(E,0f0);launchdist(A,B,C,c.h,c.w,pw,1f0,E)
   GC.@preserve distance begin;Base.unsafe_copyto!(pointer(distance),pointer(E),1;async=true);end
   CUDA.synchronize();dist=distance[1]
   if dist<best
    best=dist
    GC.@preserve bestA bestB begin
     for i in 1:c.h;Base.unsafe_copyto!(pointer(bestA,i),pointer(A,i),1;async=true);end
     for i in 1:c.w;Base.unsafe_copyto!(pointer(bestB,i),pointer(B,i),1;async=true);end
    end
   end
   push!(scores,(dist/Float32(c.h))/Float32(c.w))
  end
  CUDA.device_synchronize();total+=(time_ns()-t)*1e-9
  append!(out,Float64.(Array(A)));append!(out,Float64.(Array(B)));append!(out,Float64.(Array(E)))
  append!(out,Float64.(bestA));append!(out,Float64.(bestB));push!(out,best);append!(out,Float64.(scores))
  Array(C)==pad||error("Cmodified")
  for p in (pc,pa,pb,pe)
   bits=reinterpret(UInt32,Array(p));append!(out,Float64.(bits[1:8]));append!(out,Float64.(bits[end-7:end]))
  end
 end
 append!(prefix,out);total,prefix
end
function main()
 cs=readcases();checkfn=Libdl.dlsym(Libdl.dlopen(joinpath(@__DIR__,"study_check.so")),:study_check)
 runstudy(cs)
 for trial in 1:5
  sec,out=runstudy(cs);path="trial$(trial).bin";write(path,out);metrics=zeros(Cdouble,2)
  ret=ccall(checkfn,Cint,(Cstring,Ptr{Cdouble}),path,metrics);ret==0||error("independentscalaroracleFAIL")
  @printf("\nSTUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":true,\"checked_values\":%d,\"max_abs_error\":0,\"max_rel_error\":null,\"timing_region\":\"complete_two_restart_single_slot_factorization_sum3cases\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sec,length(out),path,length(out))
 end
end
