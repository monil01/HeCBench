using CUDA,Printf
function vmc_study(launch)
 pre=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixture.bin"))));gold=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle.bin"))));kind=read(joinpath(@__DIR__,"kinds.bin"));@assert length(pre)==9&&length(gold)==5057168&&length(kind)==length(gold)
 N=Int32(57344);B=224
 function guard(T,n)
  h=vcat(fill(T(16384),64),zeros(T,n),fill(T(16384),64));base=CuArray(h)
  (base=base,data=view(base,65:n+64))
 end
 a=ntuple(i->guard(Float32,Int(N)),7);stats=guard(Float32,4*Int(N));sums=guard(Float32,4);blocks=guard(Float32,B);states=guard(UInt32,Int(N));data=map(x->x.data,a)
 function capture!(art,full)
  for x in a;append!(art,Array(x.base));end
  full&&append!(art,Array(stats.base));append!(art,Array(states.base))
 end
 launch.init(UInt32(5551212),states.data);launch.initialize(data,states.data);launch.zero(stats.data,N);launch.propagate(Int32(1),data,stats.data,states.data,N);launch.reduce(stats.data,sums.data,blocks.data,N);CUDA.synchronize()
 for trial in 1:5
  art=copy(pre);times=Float64[]
  for ci in 0:1
   launch.init(UInt32(5551212+ci),states.data);launch.initialize(data,states.data);launch.zero(stats.data,N);CUDA.synchronize();capture!(art,false)
   launch.propagate(Int32(ci==0 ? 2 : 5),data,stats.data,states.data,N);CUDA.synchronize();capture!(art,true)
   for sample in 1:2
    CUDA.synchronize();t=time_ns();launch.zero(stats.data,N);launch.propagate(Int32(ci==0 ? 3 : 7),data,stats.data,states.data,N);launch.reduce(stats.data,sums.data,blocks.data,N);CUDA.synchronize();push!(times,(time_ns()-t)*1e-9)
    capture!(art,true);append!(art,Array(sums.base));append!(art,Array(blocks.base))
   end
  end
  @assert length(art)==length(pre)+length(gold);ok=true;ma=0.;mp=0.
  for i in eachindex(gold)
   v=art[length(pre)+i];e=abs(v-gold[i]);atol=kind[i]==1 ? 1e-12 : 2e-5;rtol=kind[i]==1 ? 2e-5 : 3e-5;ok &=isfinite(v)&&(kind[i]==2 ? v==gold[i] : e<=atol+rtol*abs(gold[i]));ma=max(ma,e);kind[i]==1&&(mp=max(mp,e))
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"timing_region\":\"sum4_source_sample_zero_propagate_and_eight_tree_reductions\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"},\"sample_times_seconds\":%s,\"wavefunction_max_abs_error\":%.17g,\"exact_RNG_accept_counts_and_guards_checked\":true,\"authored_device_kernels\":5}\n",trial,sum(times),ok ? "true" : "false",length(art),ma,path,length(art),string(times),mp)
 end
end
