using CUDA,Printf,Libdl,SHA
const PADLIB=strip(read("cpu_component_path.txt",String))
const PADHANDLE=Libdl.dlopen(PADLIB)
for (name,nargs) in ((:pad_destroy,1),(:pad_reset,1),(:pad_start,1),(:pad_join,1))
 @eval $name(ctx)=ccall(Libdl.dlsym(PADHANDLE,$(QuoteNode(name))),Cvoid,(Ptr{Cvoid},),ctx)
end
pad_data(ctx)=ccall(Libdl.dlsym(PADHANDLE,:pad_data),Ptr{Float64},(Ptr{Cvoid},),ctx)
pad_all_data(ctx)=ccall(Libdl.dlsym(PADHANDLE,:pad_all_data),Ptr{Float64},(Ptr{Cvoid},),ctx)
pad_flags(ctx)=ccall(Libdl.dlsym(PADHANDLE,:pad_flags),Ptr{Int32},(Ptr{Cvoid},),ctx)
function pad_study(launch!)
 println("STUDY_HOST_COMPONENT path=",realpath(PADLIB)," sha256=",bytes2hex(sha256(read(PADLIB))))
 words=collect(reinterpret(UInt64,read(joinpath(@__DIR__,"fixture.bin"))));@assert length(words)==163948
 cases=Any[]
 for shape in 0:1,alpha in (0f0,.1f0,.25f0,.5f0,.75f0,1f0)
  m=256;n=shape==0 ? 127 : 503;p=shape==0 ? 1 : 9;sz=m*(n+p);nt=sz÷8192;cut=trunc(Int,Float32(nt)*alpha);cap=(nt-cut)*8192;off=shape==0 ? 0 : 32768
  ctx=GC.@preserve words ccall(Libdl.dlsym(PADHANDLE,:pad_create),Ptr{Cvoid},(Cint,Cint,Cint,Cfloat,Ptr{UInt64}),m,n,p,alpha,pointer(words,off+1))
  dev=CUDA.fill(-12345.,cap+16);flags=CUDA.fill(Int32(-991827),nt+17);dv=view(dev,9:cap+8);fv=view(flags,9:nt+9)
  gold=fill(reinterpret(UInt64,-12345.),sz+16)
  for row in 0:m-1,col in 0:n+p-1;gold[9+row*(n+p)+col]=col<n ? words[off+row*n+col+1] : UInt64(0);end
  push!(cases,(m=Int32(m),n=Int32(n),p=Int32(p),size=sz,nt=Int32(nt),cut=Int32(cut),cap=cap,ctx=ctx,dev=dev,flags=flags,dv=dv,fv=fv,gold=gold))
 end
 function step(c)
  pad_reset(c.ctx)
  unsafe_copyto!(pointer(c.dv),pad_data(c.ctx),c.cap;async=false)
  unsafe_copyto!(pointer(c.fv),pad_flags(c.ctx),Int(c.nt)+1;async=false)
  launch!(c.dv,c.fv,c.n,c.m,c.p,c.nt,c.cut)
  pad_start(c.ctx);CUDA.synchronize();pad_join(c.ctx)
 end
 for c in cases;step(c);end
 for trial in 1:5
  elapsed=0.;ok=true;art=copy(words)
  for c in cases
   t=time_ns();for rep in 1:3;step(c);end;elapsed+=(time_ns()-t)*1e-9/3
   unsafe_copyto!(pad_data(c.ctx),pointer(c.dv),c.cap;async=false)
   got=copy(reinterpret(UInt64,unsafe_wrap(Vector{Float64},pad_all_data(c.ctx),c.size+16;own=false)));ok&=got==c.gold;append!(art,got)
   dg=Array(c.dev);guards=dg[vcat(1:8,c.cap+9:c.cap+16)];ok&=all(==(-12345.),guards);append!(art,reinterpret(UInt64,guards))
   hf=copy(unsafe_wrap(Vector{Int32},pad_flags(c.ctx),Int(c.nt)+1;own=false))
   for i in 0:c.nt;want=(i==0||i==c.cut ? 1 : 0)+(1<=i<=c.cut ? 1 : 0);ok&=hf[i+1]==want;end
   append!(art,UInt64.(reinterpret(UInt32,hf)))
   gf=Array(c.flags)
   for i in eachindex(gf);k=i-9;want=0<=k<=c.nt ? ((k==0||k==c.cut||k>c.cut) ? 1 : 0) : -991827;ok&=gf[i]==want;end
   append!(art,UInt64.(reinterpret(UInt32,gf)))
  end
  file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(file,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":0,\"max_rel_error\":0,\"timing_region\":\"sum12_static_hybrid_reset_copy_launch_cpuworkers_join_per_repeat\",\"jit_excluded\":true,\"internal_repeats\":3,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,ok ? "true" : "false",length(art)-length(words),file,length(art))
 end
 for c in cases;pad_destroy(c.ctx);end
end
