using CUDA,Printf
function reaction_study(diff_launch,point_launch)
 pre=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixture.bin"))));@assert length(pre)==90141 && pre[1]==4;pos=2;cases=Any[]
 function guarded(h)
  full=CuArray(vcat(fill(-12345f0,8),h,fill(-12345f0,8)))
  (full=full,p=view(full,9:length(h)+8),initial=copy(h))
 end
 for ci in 1:4
  M,steps=Int.(pre[pos:pos+1]);pos+=2;ca,cb,dt,c1,c2=Float32.(pre[pos:pos+4]);pos+=5;N=M^3
  a=Float32.(pre[pos:pos+N-1]);pos+=N;b=Float32.(pre[pos:pos+N-1]);pos+=N
  arrays=(a=guarded(a),b=guarded(b),dx2=guarded(zeros(Float32,N)),dy2=guarded(zeros(Float32,N)),dz2=guarded(zeros(Float32,N)),ra=guarded(zeros(Float32,N)),rb=guarded(zeros(Float32,N)),da=guarded(zeros(Float32,N)),db=guarded(zeros(Float32,N)))
  d=map(v->v.p,arrays);gold=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle_case$(ci).bin"))))
  @assert length(gold)==9*N
  template=CUDA.zeros(Float32,2*M)
  push!(cases,(M=Int32(M),steps=steps,N=Int32(N),ca=ca,cb=cb,dt=dt,c1=c1,c2=c2,ci=ci,arrays=arrays,d=d,gold=gold,template=template))
 end
 @assert pos==length(pre)+1
 function execute(c)
  for a in c.arrays;copyto!(a.p,a.initial);end;CUDA.synchronize()
  inputa=Array(c.d.a);inputb=Array(c.d.b);ok=reinterpret(UInt8,inputa)==reinterpret(UInt8,c.arrays.a.initial)&&reinterpret(UInt8,inputb)==reinterpret(UInt8,c.arrays.b.initial)
  t=time_ns()
  for step in 1:c.steps
   diff_launch(Val(1),c.d.a,c.d.dx2,c);diff_launch(Val(2),c.d.a,c.d.dy2,c);diff_launch(Val(3),c.d.a,c.d.dz2,c)
   point_launch(Val(:lap),c.d.da,c,c.ca)
   diff_launch(Val(1),c.d.b,c.d.dx2,c);diff_launch(Val(2),c.d.b,c.d.dy2,c);diff_launch(Val(3),c.d.b,c.d.dz2,c)
   point_launch(Val(:lap),c.d.db,c,c.cb);point_launch(Val(:reaction),c.d.da,c,0f0);point_launch(Val(:update),c.d.da,c,0f0)
  end
  CUDA.synchronize();elapsed=(time_ns()-t)*1e-9
  fields=[Array(v) for v in c.d];values=Float64[z for f in fields for z in f];error=abs.(values.-c.gold);ok &= all(isfinite,values)&&all(error .<= 1e-6 .+ 5e-6.*abs.(c.gold))&&all(>=(0f0),fields[1])&&all(>=(0f0),fields[2])
  if c.ci==3;ok &= values==c.gold;end
  mean_da=sum(Float64,fields[8])/c.N;mean_db=sum(Float64,fields[9])/c.N;ok &= abs(mean_da)<=1e-6&&abs(mean_db)<=1e-6
  if c.ci==4;ok &= all(iszero,fields[2])&&abs(sum(Float64,fields[1])/c.N-sum(Float64,inputa)/c.N)<=1e-6;end
  guards=Float64[]
  for a in c.arrays
   h=Array(a.full);g=vcat(h[1:8],h[end-7:end]);ok &= all(==(-12345f0),g);append!(guards,Float64.(g))
  end
  art=vcat(values,Float64.(inputa),Float64.(inputb),guards)
  ok,art,elapsed,maximum(error),mean_da,mean_db
 end
 for c in cases;execute(c);end
 for trial in 1:5
  art=copy(pre);times=Float64[];mas=Float64[];means=Float64[];ok=true
  for c in cases
   pass,v,t,ma,mda,mdb=execute(c);ok &=pass;append!(art,v);push!(times,t);push!(mas,ma);append!(means,[mda,mdb])
  end
  f=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(f,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"timing_region\":\"sum4_complete_zeroflux_GrayScott_timestep_batches_130_launches\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"},\"path_times_seconds\":%s,\"path_labels\":[\"nonuniform16cubed5steps\",\"central32cubed4steps\",\"equilibrium16cubed3steps\",\"diffusion16cubed1step\"],\"case_max_abs_errors\":%s,\"mean_diffusion_A_B_eachcase\":%s}\n",trial,sum(times),ok ? "true" : "false",length(art),maximum(mas),f,length(art),string(times),string(mas),string(means))
 end
end
