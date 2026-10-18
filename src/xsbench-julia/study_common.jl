using CUDA,Printf
function xsbench_study(launch,observe)
 pre=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixture.bin"))));gold=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle.bin"))));kind=read(joinpath(@__DIR__,"kinds.bin"));@assert length(kind)==length(gold)&&pre[1]==6;pos=2;cases=Any[]
 function guard(v,g)
  host=vcat(fill(g,64),v,fill(g,64));base=CuArray(host);(host=host,base=base,data=view(base,65:length(host)-64))
 end
 for ci in 1:6
  typ,I,G,N,reps,bins,M,U,IX=Int.(pre[pos:pos+8]);pos+=9;num=guard(Int32.(pre[pos:pos+11]),Int32(16384));pos+=12;concs=guard(pre[pos:pos+12*M-1],16384.);pos+=12*M;mats=guard(Int32.(pre[pos:pos+12*M-1]),Int32(16384));pos+=12*M
  points=[XPoint(pre[pos+6*i],pre[pos+6*i+1],pre[pos+6*i+2],pre[pos+6*i+3],pre[pos+6*i+4],pre[pos+6*i+5]) for i in 0:I*G-1];pos+=6*I*G;grid=guard(points,XPoint(16384,16384,16384,16384,16384,16384));energy=guard(pre[pos:pos+U-1],16384.);pos+=U;idx=guard(Int32.(pre[pos:pos+IX-1]),Int32(16384));pos+=IX
  out=guard(fill(Int32(16384),N),Int32(16384));obs=guard(fill(16384.,7*N),16384.)
  push!(cases,(typ=Int32(typ),I=Int64(I),G=Int64(G),N=Int32(N),reps=reps,bins=Int32(bins),M=Int32(M),blocks=cld(N,256),num=num,concs=concs,mats=mats,grid=grid,energy=energy,idx=idx,out=out,obs=obs))
 end
 @assert pos==length(pre)+1
 function capture!(art,a)
  for x in Array(a.base)
   if x isa XPoint;push!(art,x.energy,x.total,x.elastic,x.absorption,x.fission,x.nu_fission);else;push!(art,x);end
  end
 end
 for c in cases;launch(c);observe(c);CUDA.synchronize();end
 for trial in 1:5
  art=copy(pre);times=Float64[]
  for c in cases
   copyto!(c.out.base,c.out.host);copyto!(c.obs.base,c.obs.host);CUDA.synchronize();t=time_ns();for it in 1:c.reps;launch(c);end;CUDA.synchronize();push!(times,(time_ns()-t)*1e-9)
   observe(c);CUDA.synchronize()
   for a in (c.out,c.obs,c.num,c.concs,c.mats,c.grid,c.energy,c.idx);capture!(art,a);end
  end
  @assert length(art)==length(pre)+length(gold);ok=true;ma=0.
  for i in eachindex(gold)
   x=art[length(pre)+i];e=abs(x-gold[i]);ok &=isfinite(x)&&(kind[i]==1 ? x==gold[i] : e<=1e-10+2e-12*abs(gold[i]));ma=max(ma,e)
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"timing_region\":\"sum6_source_event_lookup_argmax_kernel_repeats_all_three_grid_methods\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"},\"path_times_seconds\":%s,\"source_derived_timed_launches\":15,\"untimed_observer_launches\":6,\"primary_authored_kernels\":1,\"scientific_observer_kernels\":1}\n",trial,sum(times),ok ? "true" : "false",length(art),ma,path,length(art),string(times))
 end
end
