from pathlib import Path
import json
R=Path('/home/imo/HeCBench_julia');P=R/'src/interval-julia';C=P/'interval-julia-cuda.jl';C.mkdir(exist_ok=True)
(P/'study_device.jl').write_text('''# Own full translation of native interval arithmetic and both stack Newton variants.
# Deliberately preserves source nearest rounding; no outward-bound guarantee.
using StaticArrays
struct IV
 lo::Float64
 hi::Float64
end
@inline IV(x::Float64)=IV(x,x)
@inline izero()=IV(NaN,NaN)
@inline iempty(x)=isnan(x.lo+x.hi)
@inline imin(x,y)=isnan(x) ? y : isnan(y) ? x : min(x,y)
@inline imax(x,y)=isnan(x) ? y : isnan(y) ? x : max(x,y)
@inline iwidth(x)=iempty(x) ? 0.0 : x.hi-x.lo
@inline imid(x)=(x.lo+x.hi)*.5
@inline iadd(x,y)=IV(x.lo+y.lo,x.hi+y.hi)
@inline isub(x,y)=IV(x.lo-y.hi,x.hi-y.lo)
@inline imul(x,y)=IV(imin(imin(x.lo*y.lo,x.lo*y.hi),imin(x.hi*y.lo,x.hi*y.hi)),imax(imax(x.lo*y.lo,x.lo*y.hi),imax(x.hi*y.lo,x.hi*y.hi)))
@inline function isquare(x)
 x.lo>=0 ? IV(x.lo*x.lo,x.hi*x.hi) : x.hi<=0 ? IV(x.hi*x.hi,x.lo*x.lo) : IV(0.,imax(x.lo*x.lo,x.hi*x.hi))
end
@inline function iintersect(x,y)
 lo=imax(x.lo,y.lo);hi=imin(x.hi,y.hi);lo<=hi ? IV(lo,hi) : izero()
end
@inline function idiv_nonzero(x,y)
 xl=y.hi<0 ? x.hi : x.lo;xu=y.hi<0 ? x.lo : x.hi
 if x.hi<0;yl=y.lo;yu=y.hi
 elseif x.lo<0;yl=y.hi<0 ? y.hi : y.lo;yu=yl
 else;yl=y.hi;yu=y.lo
 end
 IV(xl/yl,xu/yu)
end
@inline function idiv_positive(x,yu)
 x.lo==0&&x.hi==0&&return x
 x.hi<0 ? IV(-Inf,x.hi/yu) : x.lo<0 ? IV(-Inf,Inf) : IV(x.lo/yu,Inf)
end
@inline function idiv_negative(x,yl)
 x.lo==0&&x.hi==0&&return x
 x.hi<0 ? IV(x.hi/yl,Inf) : x.lo<0 ? IV(-Inf,Inf) : IV(-Inf,x.lo/yl)
end
@inline function idiv_first(x,y)
 if y.lo<=0&&y.hi>=0
  if y.lo!=0
   if y.hi!=0
    x.lo==0&&x.hi==0&&return x,false
    return x.hi<0 ? (IV(-Inf,x.hi/y.hi),true) : x.lo<0 ? (IV(-Inf,Inf),false) : (IV(-Inf,x.lo/y.lo),true)
   else;return idiv_negative(x,y.lo),false
   end
  elseif y.hi!=0;return idiv_positive(x,y.hi),false
  else;return izero(),false
  end
 else;return idiv_nonzero(x,y),false
 end
end
@inline idiv_second(x,y)=x.hi<0 ? IV(x.hi/y.lo,Inf) : IV(x.lo/y.hi,Inf)
@inline ifun(x,tid)=iadd(isquare(isub(x,IV(1.))),imul(IV(-Float64(tid)/65536.),x))
@inline ideriv(x,tid)=iadd(imul(IV(2.),x),IV(-Float64(tid)/65536.-2.))
@inline iminimal(x,tid)=!iempty(x)&&(iwidth(x)<=Float64(1f-6)*abs(imid(x))||iwidth(ifun(x,tid))<=Float64(1f-6))
@inline ibisection(x,p1,p2)=(!iempty(p1)&&iwidth(p1)>Float64(.99f0)*iwidth(x))||(!iempty(p2)&&iwidth(p2)>Float64(.99f0)*iwidth(x))
@inline function irefine(ix,tid)
 x=imid(ix);q=ifun(IV(x),tid);d=ideriv(ix,tid);p1,has=idiv_first(q,d);p1=iintersect(isub(IV(x),p1),ix);p2=has ? iintersect(isub(IV(x),idiv_second(q,d)),ix) : izero();p1,p2,has,x
end
function interval_body!(tid,out,counts,domain,choice)
 work=MVector{128,IV}(undef);tos=Int32(0);nr=Int32(0)
 if choice==0
  tos=Int32(1);@inbounds work[1]=domain
  while tos>0
   ix=@inbounds work[tos];tos-=Int32(1)
   if iminimal(ix,tid)
    @inbounds out[nr*Int32(65536)+tid+Int32(1)]=ix;nr+=Int32(1)
   else
    p1,p2,has,x=irefine(ix,tid)
    if ibisection(ix,p1,p2);p1=IV(ix.lo,x);p2=IV(x,ix.hi);has=true;end
    if !iempty(p1);tos+=Int32(1);@inbounds work[tos]=p1;end
    if has&&!iempty(p2);tos+=Int32(1);@inbounds work[tos]=p2;end
   end
  end
 else
  ix=domain
  while true
   p1,p2,has,x=irefine(ix,tid)
   if iminimal(p1,tid);@inbounds out[nr*Int32(65536)+tid+Int32(1)]=p1;nr+=Int32(1);p1=izero();end
   if has&&iminimal(p2,tid);@inbounds out[nr*Int32(65536)+tid+Int32(1)]=p2;nr+=Int32(1);p2=izero();end
   if ibisection(ix,p1,p2);p1=IV(ix.lo,x);p2=IV(x,ix.hi);has=true;end
   if !iempty(p1)
    ix=p1
    if has&&!iempty(p2);tos+=Int32(1);@inbounds work[tos]=p2;end
   elseif has&&!iempty(p2);ix=p2
   elseif tos==0;break
   else;ix=@inbounds work[tos];tos-=Int32(1)
   end
  end
 end
 @inbounds counts[tid+Int32(1)]=nr
 nothing
end
@inline function interval_echo_body!(tid,echo,domain)
 a=ifun(IV(1.),tid);@inbounds echo[Int32(2)*tid+Int32(1)]=a.lo;@inbounds echo[Int32(2)*tid+Int32(2)]=a.hi
 if tid==0
  @inbounds echo[131073]=domain.lo;@inbounds echo[131074]=domain.hi;@inbounds echo[131075]=1.+2.0^-53;@inbounds echo[131076]=1.+3.*2.0^-53
 end
 nothing
end
''')
(C/'main.jl').write_text('''using CUDA
const MODEL="cuda_jl"
include("../study_device.jl")
function interval_kernel!(out,counts,domain,choice)
 tid=(CUDA.blockIdx().x-Int32(1))*Int32(64)+CUDA.threadIdx().x-Int32(1)
 interval_body!(tid,out,counts,domain,choice);nothing
end
function echo_kernel!(out,domain)
 tid=(CUDA.blockIdx().x-Int32(1))*Int32(64)+CUDA.threadIdx().x-Int32(1)
 interval_echo_body!(tid,out,domain);nothing
end
function launch_main(s,choice)
 CUDA.@cuda threads=64 blocks=1024 interval_kernel!(s.result.data,s.counts.data,s.domain,Int32(choice));nothing
end
function launch_echo(s)
 CUDA.@cuda threads=64 blocks=1024 echo_kernel!(s.echo.data,s.domain);nothing
end
include("../study_common.jl")
main()
''')
(P/'study_common.jl').write_text('''const ROOT=@__DIR__
const CHECKLIB=joinpath(ROOT,"study_check.so")
function alloc(::Type{T},n)where T
 storage=CuArray{T}(undef,n+64);(storage=storage,data=view(storage,33:n+32),n=n)
end
function reset!(s)
 for b in(s.result,s.counts,s.echo);CUDA.memset(convert(CUDA.CuPtr{UInt8},pointer(b.storage)),UInt8(165),length(b.storage)*sizeof(eltype(b.storage)));end
 CUDA.device_synchronize();nothing
end
function run_main(s,ch)
 CUDA.device_synchronize();start=time_ns();launch_main(s,ch);CUDA.device_synchronize();(time_ns()-start)*1e-9
end
function capture(s,folder,trial,ch)
 result=Array(s.result.storage);counts=Array(s.counts.storage);echo=Array(s.echo.storage);missing=Ref{Cint}(0);extra=Ref{Cint}(0)
 ok=GC.@preserve result counts ccall((:interval_check_coverage,CHECKLIB),Cint,(Ptr{Cvoid},Ptr{Int32},Ref{Cint},Ref{Cint}),pointer(result,33),pointer(counts,33),missing,extra)==1
 println(stderr,"trial",trial," choice",ch," exactrootcoverage_missing",missing[]," extra_candidates",extra[])
 guards=UInt8[]
 for a in(result,counts,echo);v=reinterpret(UInt8,a);n=32*sizeof(eltype(a));append!(guards,view(v,1:n));append!(guards,view(v,length(v)-n+1:length(v)));end
 ok&=all(==(UInt8(165)),guards);values=Float64[]
 for i in 0:65535
  k=Int(counts[33+i]);push!(values,k);if !(0<=k<=128);ok=false;continue;end
  intervals=[(result[33+j*65536+i].lo,result[33+j*65536+i].hi)for j in 0:k-1]
  sort!(intervals)
  for (lo,up)in intervals;push!(values,lo);push!(values,up);ok&=isfinite(lo)&&isfinite(up)&&lo<=up;end
 end
 path=joinpath(folder,"$(MODEL)_$(getpid())_trial$(trial)_choice$(ch)")
 for(suffix,a)in((".intervals.bin",result),(".counts.i32",counts),(".echo.f64",echo),(".guards.u8",guards));open(io->write(io,a),path*suffix,"w");end
 println("STUDY_RAW_ARTIFACT {\\"trial\\":",trial,",\\"choice\\":",ch,",\\"path\\":\\"",path,"\\",\\"result_bytes\\":",sizeof(IV)*length(result),",\\"guard_bytes\\":",length(guards),"}")
 values,ok
end
function main()
 length(ARGS)==1||error("persistent raw folder required");folder=ARGS[1];mkpath(folder);sizeof(IV)==16||error("interval ABI");s=(result=alloc(IV,65536*128),counts=alloc(Int32,65536),echo=alloc(Float64,2*65536+4),domain=IV(Float64(.01f0),4.))
 for ch in 0:1;reset!(s);run_main(s,ch);launch_echo(s);CUDA.device_synchronize();end
 for trial in 1:5
  prefix=Float64[];values=Float64[];seconds=0.;ok=true
  for ch in 0:1
   reset!(s);launch_echo(s);CUDA.device_synchronize();e=Array(s.echo.data);append!(prefix,e);append!(prefix,Float64[ch,65536,1,1024,64]);seconds+=run_main(s,ch);out,pass=capture(s,folder,trial,ch);append!(values,out);ok&=pass
  end
  append!(prefix,values);path="trial$(trial).bin";open(io->write(io,prefix),path,"w")
  println("STUDY_TRIAL {\\"trial\\":",trial,",\\"time_seconds\\":",seconds,",\\"correct\\":",ok,",\\"checked_values\\":",length(prefix),",\\"max_abs_error\\":null,\\"max_rel_error\\":null,\\"timing_region\\":\\"sum2_full65536equation_Newton_variants_kernel_and_contextcompletion\\",\\"jit_excluded\\":true,\\"internal_repeats\\":1,\\"output_artifact\\":{\\"path\\":\\"",path,"\\",\\"dtype\\":\\"float64\\",\\"count\\":",length(prefix),",\\"byte_order\\":\\"little\\"}}")
  ok||error("independent rootcoverage/shape/guard failed")
 end
end
''')
d=json.loads((P/'study_config_native.json').read_text());d['reuse_result']=str(S/'runs/20261019T214817896330Z_interval/result.json');d['reuse_models']=['cuda'];d['models']['cuda_jl']={'structured':True,'cwd':str(P),'bind_all_declared_dependencies':True,'source_dependencies':[str(P/x)for x in ['study_device.jl','study_common.jl','study_check.cpp','study_check.h','study_check.so','SCOPE.md','candidate_contract_review.md','rounding_witness.json']],'command':['/home/imo/.juliaup/bin/julia','--project='+str(R/'src/_julia_ka_env'),str(C/'main.jl'),str(S/'interval_raw_qualified')],'verification_method':'all65536exactrootcoverage_complete_raw_candidateintervals_guards_deviceecho'};(P/'study_config_refs.json').write_text(json.dumps(d,indent=2)+'\n');print('Full ownCUDAjl intervalarithmetic/bothNewtonstackvariants/commoncapture pipeline written')
