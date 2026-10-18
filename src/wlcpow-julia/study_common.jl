using CUDA,Printf
function wlcpow_study(launch)
 pre=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixture.bin"))));gold=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle.bin"))));kinds=read(joinpath(@__DIR__,"kinds.bin"));@assert length(pre)==16648&&length(gold)==35969&&length(kinds)==length(gold);pos=2;cases=Any[]
 function guarded(values,guard)
  host=vcat(fill(guard,64),values,fill(guard,64));base=CuArray(host);(host=host,base=base,data=view(base,65:length(host)-64))
 end
 function getvec4(n)
  vals=[WVec4(Float32(pre[pos+4*i]),Float32(pre[pos+4*i+1]),Float32(pre[pos+4*i+2]),Float32(pre[pos+4*i+3])) for i in 0:n-1];pos+=4*n;guarded(vals,WVec4(16384,16384,16384,16384))
 end
 for ci in 1:5
  N,nt,reps=Int.(pre[pos:pos+2]);period=Tuple(Float32.(pre[pos+3:pos+5]));pos+=6;L=N+1;E=2*N+1;co=getvec4(L);vel=getvec4(L);nb=guarded(Int32.(pre[pos:pos+L-1]),Int32(16384));pos+=L;bo=[WInt2(Int32(pre[pos+2*i]),Int32(pre[pos+2*i+1])) for i in 0:E-1];pos+=2*E;bonds=guarded(bo,WInt2(16384,16384));l0=guarded(pre[pos:pos+E-1],16384.);pos+=E;pars=Any[]
  for t in 1:8;push!(pars,guarded(Float32.(pre[pos:pos+nt]),16384f0));pos+=nt+1;end
  fs=Any[]
  for a in 1:3;push!(fs,guarded(pre[pos:pos+L-1],16384.));pos+=L;end
  push!(cases,(N=Int32(N),nt=Int32(nt),reps=reps,period=period,blocks=cld(N,128),co=co,vel=vel,nb=nb,bonds=bonds,l0=l0,params=Tuple(pars),force=Tuple(fs)))
 end
 @assert pos==length(pre)+1
 function capture!(art,a)
  for x in Array(a.base)
   if x isa WVec4;push!(art,x.x,x.y,x.z,x.w);elseif x isa WInt2;push!(art,x.x,x.y);else;push!(art,x);end
  end
 end
 for c in cases;launch(c);CUDA.synchronize();end
 for trial in 1:5
  art=copy(pre);times=Float64[]
  for c in cases
   for f in c.force;copyto!(f.base,f.host);capture!(art,f);end
   CUDA.synchronize();t=time_ns();for _ in 1:c.reps;launch(c);end;CUDA.synchronize();push!(times,(time_ns()-t)*1e-9)
   for f in c.force;capture!(art,f);end
   for a in (c.co,c.vel,c.nb,c.bonds,c.l0,c.params...);capture!(art,a);end
  end
  @assert length(art)==length(pre)+length(gold);ok=true;ma=0.
  for i in eachindex(gold)
   x=art[length(pre)+i];err=abs(x-gold[i]);ok &=isfinite(x)&&(kinds[i]==1 ? x==gold[i] : err<=1e-4+2e-4*abs(gold[i]));ma=max(ma,err)
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"timing_region\":\"sum5_WLC_power_viscous_TEA_force_repeated_accumulation\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"},\"path_times_seconds\":%s,\"source_derived_kernel_launches\":12,\"authored_device_kernels\":1}\n",trial,sum(times),ok ? "true" : "false",length(art),ma,path,length(art),string(times))
 end
end
