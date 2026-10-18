using Printf
include(joinpath(@__DIR__,"study_cases.jl"))
const G=8
const GUARD=1048575f0
const POISON=-12345.5f0
function study(allocate,launch,transpose_launch)
 f=collect(reinterpret(Float32,read(joinpath(@__DIR__,"input_f32.bin"))));oracle=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle_f64.bin"))));states=[]
 for c in CASES
  z=c.n*c.m;host=[vcat(fill(GUARD,G),k<=4 ? f[c.io+(k-1)*z+1:c.io+k*z] : fill(POISON,z),fill(GUARD,G))for k in 1:7];raw=[allocate(h)for h in host];dev=[view(a,G+1:length(a)-G)for a in raw];push!(states,(c=c,host=host,raw=raw,dev=dev))
 end
 function phase(operation)
  operation();CUDA.device_synchronize();start=time_ns()
  for r in 1:3;operation();end
  CUDA.device_synchronize();(time_ns()-start)*1e-9/3
 end
 function execute(s)
  c=s.c;n=c.n;m=c.m;mode=c.mode;ptrs=[1,2,3,4,5];tmp=6;re=mode in (5,7,8);stages=Float64[]
  if re
   for k in 1:4
    src=ptrs[k];dst=tmp;push!(stages,phase(()->transpose_launch(s.dev[dst],s.dev[src],n,m)));ptrs[k],tmp=tmp,ptrs[k]
   end
  end
  push!(stages,phase(()->launch(mode,s.dev[ptrs[1]],s.dev[ptrs[2]],s.dev[ptrs[3]],s.dev[ptrs[4]],s.dev[ptrs[5]],s.dev[7],n,m)))
  if re
   src=ptrs[5];dst=tmp;push!(stages,phase(()->transpose_launch(s.dev[dst],s.dev[src],m,n)));ptrs[5],tmp=tmp,ptrs[5]
  end
  stages
 end
 function capture!(payload,s)
  for a in s.raw;append!(payload,Float64.(Array(a)));end
 end
 function check(s)
  c=s.c;n=c.n;m=c.m;z=n*m;re=c.mode in (5,7,8);ok=true;maxerr=0.;maxback=0.
  for k in 1:7
   v=Array(s.raw[k]);ok&=all(==(GUARD),v[1:G])&&all(==(GUARD),v[end-G+1:end]);numeric=(!re&&k==5)||(re&&k in (4,5))||(k==7&&c.mode in (4,5,8));want=oracle[c.oo+(k-1)*z+1:c.oo+k*z]
   for j in 1:z
    value=Float64(v[j+G]);err=abs(value-want[j]);maxerr=max(maxerr,err);ok&=isfinite(value)&&(numeric ? err<=2e-5+2e-4*abs(want[j]) : value==want[j])
   end
  end
  x=Array(s.raw[re ? 4 : 5])
  for sys in 0:m-1
   an=0.;xn=0.;bn=0.;rn=0.
   for i in 0:n-1
    j=sys*n+i+G+1;aa=Float64(s.host[1][j]);bb=Float64(s.host[2][j]);cc=Float64(s.host[3][j]);rhs=Float64(s.host[4][j]);xx=Float64(x[j]);res=bb*xx-rhs
    if i>0;res+=aa*Float64(x[j-1]);end
    if i+1<n;res+=cc*Float64(x[j+1]);end
    rn=max(rn,abs(res));an=max(an,abs(aa)+abs(bb)+abs(cc));xn=max(xn,abs(xx));bn=max(bn,abs(rhs))
   end
   back=rn/(an*xn+bn);maxback=max(maxback,back);ok&=isfinite(back)&&back<=1e-5
  end
  ok,maxerr,maxback
 end
 for trial in 1:5
  pre=Float64[];post=Float64[];ok=true;seconds=0.;maxerr=0.;maxback=0.
  for s in states
   for k in 1:7;copyto!(s.raw[k],s.host[k]);end
   CUDA.device_synchronize();capture!(pre,s);stages=execute(s);seconds+=sum(stages);good,err,back=check(s);ok&=good;maxerr=max(maxerr,err);maxback=max(maxback,back);capture!(post,s)
   println("STUDY_CASE trial=",trial," case=",s.c.case_id," mode=",s.c.mode," stage_seconds=",join(stages,','))
  end
  append!(pre,post);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"tridiagonal_trial$(trial).bin");open(io->write(io,pre),path,"w")
  @printf("STUDY_TRIAL {\"trial\":%d,\"correct\":%s,\"jit_excluded\":true,\"time_seconds\":%.12g,\"internal_repeats\":3,\"timing_region\":\"sum45_original_solver_paths_and_individual_reorder_stages_warm_then_three_call_batch_completion\",\"max_abs_error\":%.12g,\"max_normwise_backward_error\":%.12g,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"shape\":[%d],\"byte_order\":\"little\"}}\n",trial,ok ? "true" : "false",seconds,maxerr,maxback,path,length(pre),length(pre))
  ok||error("Full tridiagonal solution/workspace/oracle/guard/backward-error contract")
 end
end
