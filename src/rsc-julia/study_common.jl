using CUDA,Printf
struct Flow
 x::Int32;y::Int32;vx::Int32;vy::Int32
end
const FIT_LIB=joinpath(@__DIR__,"../../.porting-state/porting_logs/rsc-julia-shared/study_fit.so")
const GUARD=Int32(16384)
function guarded(::Type{T},n) where T
 reset=fill(T(GUARD),n+16);base=CuArray(reset)
 (base=base,data=view(base,9:n+8),reset=reset,n=n)
end
function rsc_study(launch,model)
 pre=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixture.bin"))));gold=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle.bin"))));@assert length(pre)==3061&&pre[1]==4&&length(gold)==6756
 pos=2;cases=Any[]
 for ci in 1:4
  n,m,e=Int.(pre[pos:pos+2]);conv=Float32(pre[pos+3]);cpu,b,t=Int.(pre[pos+4:pos+6]);pos+=7
  flat=Int32.(pre[pos:pos+4*n-1]);pos+=4*n;flows=collect(reinterpret(Flow,flat));pairs=Int32.(pre[pos:pos+2*m-1]);pos+=2*m
  p=guarded(Float32,4*m);f=guarded(Int32,4*n);c=guarded(Int32,m);o=guarded(Int32,m);id=guarded(Int32,1);f.reset[9:8+4*n]=flat;copyto!(f.base,f.reset)
  push!(cases,(n=Int32(n),m=Int32(m),e=Int32(e),conv=conv,cpu=Int32(cpu),blocks=b,threads=t,flows=flows,flat=flat,pairs=pairs,params=zeros(Float32,4*m),hc=zeros(Int32,m),ho=zeros(Int32,m),hostid=zeros(Int32,1),p=p,f=f,c=c,o=o,id=id,gflows=view(reinterpret(Flow,f.base),3:n+2)))
 end
 @assert pos==length(pre)+1&&sizeof(Flow)==16
 function execute(c)
  stages=Float64[];sizehint!(stages,6);start=time_ns();mark=start
  fill!(c.params,0);fill!(c.hc,0);fill!(c.ho,0);c.hostid[1]=0
  copyto!(c.p.reset,9,c.params,1,4*c.m);copyto!(c.c.reset,9,c.hc,1,c.m);copyto!(c.o.reset,9,c.ho,1,c.m);c.id.reset[9]=0
  for q in (c.p,c.c,c.o,c.id);copyto!(q.base,q.reset);end
  now=time_ns();push!(stages,(now-mark)*1e-9);mark=now
  ccall((:rsc_fit,FIT_LIB),Cvoid,(Ptr{Float32},Ptr{Flow},Cint,Ptr{Int32},Cint,Cint,Cfloat,Ptr{Int32},Cint),c.params,c.flows,c.n,c.pairs,c.m,c.e,c.conv,c.hostid,c.cpu)
  now=time_ns();push!(stages,(now-mark)*1e-9);mark=now
  copyto!(c.p.data,c.params)
  now=time_ns();push!(stages,(now-mark)*1e-9);mark=now
  launch(c);CUDA.synchronize()
  now=time_ns();push!(stages,(now-mark)*1e-9);mark=now
  copyto!(c.hostid,c.id.data);n=Int(c.hostid[1]);@assert 0<=n<=c.m
  copyto!(c.hc,1,c.c.data,1,n);copyto!(c.ho,1,c.o.data,1,n)
  now=time_ns();push!(stages,(now-mark)*1e-9);mark=now
  best=Int32(-1);minimum=c.n
  for i in 1:n
   if c.ho[i]<minimum;minimum=c.ho[i];best=c.hc[i];end
  end
  now=time_ns();push!(stages,(now-mark)*1e-9);secs=(time_ns()-start)*1e-9
  p,f,ca,o,id=map(q->Array(q.base),(c.p,c.f,c.c,c.o,c.id));candidates=ca[9:8+n];counts=o[9:8+n];valid=n==0 ? best==-1&&minimum==c.n : any(i->candidates[i]==best&&counts[i]==minimum,1:n)
  ok=valid&&length(unique(candidates))==n&&all(x->0<=x<c.m,candidates)&&all(x->0<=x<=c.n,counts)&&all(iszero,ca[9+n:8+c.m])&&all(iszero,o[9+n:8+c.m])&&id[9]==n
  ok &=f[9:8+4*c.n]==c.flat&&p[9:8+4*c.m]==c.params
  perm=sortperm(candidates);canonical=Float64[]
  for i in perm;push!(canonical,candidates[i],counts[i]);end
  append!(canonical,fill(-1.,2*(c.m-n)))
  guards=Float64[]
  for (a,q) in zip((p,f,ca,o,id),(c.p,c.f,c.c,c.o,c.id));append!(guards,a[1:8]);append!(guards,a[end-7:end]);end
  ok &=all(==(GUARD),guards)
  vals=vcat(Float64[n,minimum,valid ? 1 : 0],Float64.(c.params),canonical,Float64.(f[9:8+4*c.n]),Float64.(p[9:8+4*c.m]),guards)
  raw=Float64[n,best,minimum,valid ? 1 : 0];append!(raw,c.params)
  for i in 1:c.m;push!(raw,ca[8+i],o[8+i]);end
  append!(raw,f[9:8+4*c.n]);append!(raw,p[9:8+4*c.m]);append!(raw,guards)
  (vals=vals,raw=raw,secs=secs,stages=stages,ok=ok)
 end
 for c in cases;@assert execute(c).ok;end
 rawdir=joinpath(@__DIR__,"../../.porting-state/porting_logs/rsc-julia-shared/raw",model);mkpath(rawdir)
 for trial in 1:5
  art=copy(pre);raw=Float64[];paths=Float64[];stages=Float64[];ok=true
  for c in cases
   z=execute(c);append!(art,z.vals);append!(raw,z.raw);push!(paths,z.secs);append!(stages,z.stages);ok &=z.ok
  end
  ok &=art[length(pre)+1:end]==gold
  output=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");rp=joinpath(rawdir,"raw_trial$(trial)_pid$(getpid()).bin");write(output,art);write(rp,raw)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":null,\"max_rel_error\":null,\"timing_region\":\"four_hybrid_RANSAC_passes_reset_fit_copy_evaluate_read_select\",\"jit_excluded\":true,\"internal_repeats\":1,\"numerical_agreement_claim\":false,\"raw_output_identity_claim\":false,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"},\"raw_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d},\"path_times_seconds\":%s,\"stage_times_seconds\":%s}\n",trial,sum(paths),ok ? "true" : "false",length(art),output,length(art),rp,length(raw),string(paths),string(stages))
 end
end
