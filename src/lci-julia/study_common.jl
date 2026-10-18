using Printf
const G=16;const GUARD=12345.5
function run_study(launch)
 base=@__DIR__;inputs=collect(reinterpret(Float64,read(joinpath(base,"inputs_f64.bin"))));alphah=collect(reinterpret(Float64,read(joinpath(base,"alpha_f64.bin"))));gold=collect(reinterpret(Float64,read(joinpath(base,"oracle_f64.bin"))))
 guarded(v)=vcat(fill(GUARD,G),v,fill(GUARD,G))
 cs=[CuArray(guarded(inputs[(q-1)*65+1:q*65])) for q in 1:4];rs=[CuArray(guarded(fill(GUARD,65))) for q in 1:4];aa=CuArray(guarded(alphah));alpha=view(aa,G+1:G+193);times=(.2,1.,200.,17.5)
 prefix=vcat([Array(c) for c in cs]...,[Array(r) for r in rs]...,Array(aa))
 for q in 1:4;launch(view(rs[q],G+1:G+65),view(cs[q],G+1:G+65),alpha,times[q]);end;CUDA.device_synchronize()
 for trial in 1:5
  for q in 1:4;fill!(rs[q],GUARD);end;CUDA.device_synchronize();seconds=Float64[]
  for q in 1:4
   t0=time_ns();launch(view(rs[q],G+1:G+65),view(cs[q],G+1:G+65),alpha,times[q]);CUDA.device_synchronize();push!(seconds,(time_ns()-t0)*1e-9)
  end
  correct=true;maxerr=0.;outs=Float64[]
  for q in 1:4
   r=Array(rs[q]);x=r[G+1:G+64];ref=gold[(q-1)*64+1:q*64];err=abs.(x.-ref);correct &= all(isfinite,x)&&all(err .<=1e-8 .+1e-10.*abs.(ref));maxerr=max(maxerr,maximum(err));correct &= all(==(GUARD),r[vcat(1:G,G+65:G+65+G)]) && Array(cs[q])==prefix[(q-1)*97+1:q*97];append!(outs,r)
  end
  correct &= Array(aa)==prefix[end-224:end]
  file=tempname()*".bin";open(file,"w") do io;write(io,vcat(prefix,outs));end
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"max_abs_error\":%.12g,\"max_rel_error\":null,\"checked_values\":256,\"jit_excluded\":true,\"timing_region\":\"sum_four_single_kernel_host_launch_plus_device_completion\",\"component_time_seconds\":[%s],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"shape\":[%d],\"byte_order\":\"little\"}}\n",trial,sum(seconds),string(correct),maxerr,join(seconds,","),file,length(prefix)+length(outs),length(prefix)+length(outs))
 end
end
