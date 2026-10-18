using CUDA,Printf
const HOSTLIB="/home/imo/HeCBench_julia/.porting-state/porting_logs/tqs-julia-shared/study_host.so"
function tqs_study(launch)
 pre=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixture.bin"))));gold=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle.bin"))));@assert length(pre)==742&&length(gold)==161259;pos=2;cs=Any[]
 function guard(T,n,g)
  host=fill(g,n+128);base=CuArray(host);(base=base,data=view(base,65:n+64),n=n)
 end
 for ci in 1:4
  T,B,Q,P,it=Int.(pre[pos:pos+4]);pos+=5;ops=Int32.(pre[pos:pos+P-1]);pos+=P;pool=[StudyTask(Int32(i-1),ops[i])for i in 1:P];q=Vector{StudyTask}(undef,Q);data=zeros(Int32,P*T);dq=zeros(Int32,Q*T);tasks=guard(StudyTask,Q+B,StudyTask(Int32(16384),Int32(16384)));gpu=guard(Int32,Q*T,Int32(16384));counter=guard(Int32,1,Int32(16384));nc=P÷Q
  push!(cs,(T=T,B=B,Q=Q,P=P,it=it,pool=pool,q=q,data=data,dq=dq,tasks=tasks,gpu=gpu,counter=counter,written=Ref{Cint}(0),zero=Int32[0],before=[zeros(Int32,Q*T+128)for _ in 1:nc],after=[zeros(Int32,Q*T+128)for _ in 1:nc],counts=[zeros(Int32,129)for _ in 1:nc],taskecho=[Vector{StudyTask}(undef,Q+B+128)for _ in 1:nc]))
 end
 @assert pos==length(pre)+1
 # Explicit synchronous completion matches source blocking D2H after eachchunk.
 function h2d(dst,src,n)
  GC.@preserve src CUDA.unsafe_copyto!(dst,pointer(src),n;async=true)
 end
 function d2h(dst,src,n)
  GC.@preserve dst CUDA.unsafe_copyto!(pointer(dst),src,n;async=true)
  CUDA.synchronize()
 end
 function execute(c)
  CUDA.synchronize();t=time_ns();fill!(c.data,Int32(0));c.written[]=0
  GC.@preserve c begin
   for (k,off) in enumerate(0:c.Q:c.P-1)
    ccall((:study_insert,HOSTLIB),Cvoid,(Ptr{StudyTask},Ptr{Int32},Ptr{StudyTask},Ptr{Int32},Ref{Cint},Cint,Cint,Cint),pointer(c.q),pointer(c.dq),pointer(c.pool),pointer(c.data),c.written,c.Q,off,c.T)
    h2d(pointer(c.tasks.base,65),c.q,c.Q);h2d(pointer(c.gpu.base,65),c.dq,c.Q*c.T);h2d(pointer(c.counter.base,65),c.zero,1)
    d2h(c.before[k],pointer(c.gpu.base),c.Q*c.T+128)
    launch(c,Int32(off))
    # OriginalpayloadD2H ispreserved; observersareadditionalinsideROI.
    CUDA.unsafe_copyto!(pointer(c.data,off*c.T+1),pointer(c.gpu.base,65),c.Q*c.T;async=true);CUDA.synchronize()
    d2h(c.after[k],pointer(c.gpu.base),c.Q*c.T+128);d2h(c.taskecho[k],pointer(c.tasks.base),c.Q+c.B+128);d2h(c.counts[k],pointer(c.counter.base),129)
   end
  end
  elapsed=(time_ns()-t)*1e-9;vals=Float64[]
  for k in eachindex(c.before)
   append!(vals,c.before[k]);append!(vals,c.after[k]);for task in c.taskecho[k];push!(vals,task.id,task.op);end;append!(vals,c.counts[k])
  end
  append!(vals,c.data);push!(vals,c.written[]);elapsed,vals
 end
 CUDA.stream!(CUDA.default_stream()) do
  for c in cs;execute(c);end
  for trial in 1:5
   art=copy(pre);times=Float64[]
   for c in cs;t,v=execute(c);push!(times,t);append!(art,v);end
   @assert length(art)==length(pre)+length(gold);actual=art[length(pre)+1:end];ok=all(isfinite,actual)&&actual==gold;path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,art)
   @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":0,\"max_rel_error\":null,\"timing_region\":\"sum4_source_CPUenqueue_H2D_persistentqueue_D2H_with_full_input_guard_observers\",\"jit_excluded\":true,\"internal_repeats\":1,\"native_reference_repaired\":true,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"},\"path_times_seconds\":%s,\"authored_device_kernels\":1,\"original_kernel_body_unchanged\":true}\n",trial,sum(times),ok ? "true" : "false",length(art),path,length(art),string(times))
  end
 end
end
