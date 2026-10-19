const BN_G=32
function bn_study(launch_score!,launch_compute!)
 root=@__DIR__;fixture=reinterpret(Float64,read(joinpath(root,"fixture.bin")));oracle=reinterpret(Float64,read(joinpath(root,"score_oracle.bin")));mathgold=reinterpret(Float64,read(joinpath(root,"math_score_oracle.bin")));mathbound=reinterpret(Float64,read(joinpath(root,"math_score_budget.bin")))
 lib=Libdl.dlopen(joinpath(pwd(),"libbn_host.so"));sym(name)=Libdl.dlsym(lib,name);libraryhash=bytes2hex(SHA.sha256(read(joinpath(pwd(),"libbn_host.so"))))
 function guarded(v)
  sentinel=eltype(v)(eltype(v)==UInt8 ? 173 : -37)
  return vcat(fill(sentinel,32),v,fill(sentinel,32))
 end
 for trial in 0:5
  prefix=Float64[3];computed=Float64[];extra=Float64[];off=2;co=1;ok=true;secs=0.;gens=0.;search=0.;maxerr=0.;matherr=0.
  for ci in 0:2
   ccall(sym(:bn_reset),Cvoid,(Cint,),ci);head=copy(fixture[off:off+3]);off+=4;N,D,states,S=Int32.(head);append!(prefix,head)
   raw=Int32.(fixture[off:off+N*D-1]);off+=N*D;lg=Float32.(fixture[off:off+D+1]);off+=D+2
   # Canonical serializedtable bits override compiler-dependent hostlog overload.
   @assert raw==unsafe_wrap(Vector{Int32},ccall(sym(:bn_data),Ptr{Int32},()),N*D;own=false)
   copyto!(unsafe_wrap(Vector{Float32},ccall(sym(:bn_lg),Ptr{Float32},()),D+2;own=false),lg)
   inputs=Any[guarded(raw),guarded(lg),guarded(zeros(Float32,N*S)),guarded(zeros(UInt8,N)),guarded(zeros(Float32,1)),guarded(zeros(Int32,4))]
   base=CuArray.(inputs);buffers=map(a->view(a,33:length(a)-32),base);data,gamma,locals,mask,dscores,parents=buffers
   for (b,h)in zip(base,inputs);a=Array(b);ok&=a==h;append!(prefix,a);end
   hmask=zeros(UInt8,N);hscores=zeros(Float32,1);hparents=zeros(Int32,4);snapshot=zeros(Float64,573)
   CUDA.synchronize(CUDA.context());start=time_ns();fill!(locals,0f0);launch_score!(S,locals,data,gamma);CUDA.synchronize(CUDA.context());dt=(time_ns()-start)*1e-9;secs+=dt;gens+=dt
   actual=Array(locals)
   for i in eachindex(actual)
    e=abs(Float64(actual[i])-oracle[co+i-1]);maxerr=max(maxerr,e);matherr=max(matherr,abs(Float64(actual[i])-mathgold[co+i-1]));ok&=isfinite(actual[i])&&e<=1e-3+2e-6*abs(oracle[co+i-1])&&abs(Float64(actual[i])-mathgold[co+i-1])<=mathbound[co+i-1]
   end
   append!(computed,actual);co+=N*S
   for iteration in 1:100
    ccall(sym(:bn_prepare_iteration),Cvoid,())
    for node in Int32(0):N-Int32(1)
     total=ccall(sym(:bn_total),Cint,(Cint,),node);blocks=div(total,256)+1;@assert blocks==1
     source_mask=unsafe_wrap(Vector{UInt8},ccall(sym(:bn_mask),Ptr{UInt8},(Cint,),node),N;own=false)
     CUDA.synchronize(CUDA.context());start=time_ns();fill!(parents,Int32(0));fill!(dscores,reinterpret(Float32,UInt32(0xc1c1c1c1)));copyto!(mask,source_mask);CUDA.synchronize(CUDA.context());dt=(time_ns()-start)*1e-9;secs+=dt;search+=dt
     # Exact actualmask echo, outside both matched measuredsegments.
     actualmask=Array(base[4]);ok&=actualmask[33:end-32]==source_mask;append!(prefix,actualmask)
     CUDA.synchronize(CUDA.context());start=time_ns();launch_compute!(S,locals,mask,node,Int32(total),dscores,parents);copyto!(hscores,dscores);copyto!(hparents,parents)
     # Blockingreadback before consuming host pointers; exact matching completion.
     CUDA.synchronize(CUDA.context())
     GC.@preserve hscores hparents ccall(sym(:bn_accept_node),Cvoid,(Cint,Cint,Ptr{Float32},Ptr{Int32}),node,blocks,hscores,hparents)
     CUDA.synchronize(CUDA.context());dt=(time_ns()-start)*1e-9;secs+=dt;search+=dt
     append!(computed,hscores);append!(computed,hparents)
    end
    ccall(sym(:bn_finish_iteration),Cvoid,());GC.@preserve snapshot ccall(sym(:bn_snapshot),Cvoid,(Ptr{Float64},),snapshot);append!(computed,snapshot)
   end
   for (b,h)in zip(base,inputs)
    a=Array(b);ok&=a[1:32]==h[1:32]&&a[end-31:end]==h[end-31:end];append!(extra,a)
   end
   ok&=Array(base[1])==inputs[1]&&Array(base[2])==inputs[2]
   for b in base;CUDA.unsafe_free!(b);end
   ccall(sym(:bn_release),Cvoid,())
  end
  @assert off==length(fixture)+1&&co==length(oracle)+1
  if trial==0;ok||error("BNwarmup CPUoracle mismatch maxabs=$maxerr math=$matherr");continue;end
  data=vcat(prefix,computed,extra);@assert length(prefix)==220714&&length(data)==434116
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,data)
  # Fullindependent mathematical+DAG/argmax/feedback certificate BEFORE acceptedrow.
  ok&=success(pipeline(Cmd(["python3",joinpath(root,"audit_native_outputs.py"),path]),stdout=devnull,stderr=stderr))
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"genScore_seconds\":%.17g,\"search_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"independent_math_max_abs_error\":%.17g,\"host_library_sha256\":\"%s\",\"timing_region\":\"sum3cases_source_scoreplus_segmented100order_search_resets_transfers_GPUreduction_CPUselection_completion\",\"jit_excluded\":true,\"internal_repeats\":1,\"native_reference_repaired\":true,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,secs,gens,search,ok ? "true" : "false",length(data),maxerr,matherr,libraryhash,path,length(data))
  ok||error("IndependentFullGraphCertificateFailed")
 end
end
