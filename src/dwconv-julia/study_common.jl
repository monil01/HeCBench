using CUDA,Printf
const ROOT=@__DIR__
struct ConvCase
 id::Int;m::Int;k::Int;n::Int;c::Int;h::Int;w::Int;oc::Int;oh::Int;ow::Int;ni::Int;nw::Int;nb::Int;no::Int
end
function makecase(id,m,k)
 n=2;c=3;h=11;w=13;oc=c*m;oh=h+3-k;ow=w+3-k
 ConvCase(id,m,k,n,c,h,w,oc,oh,ow,n*c*h*w,oc*k*k,oc,n*oc*oh*ow)
end
function dwconv_study(factory)
 pairs=[(m,k) for m in 1:4 for k in (1,3,5)]
 prepared=map(enumerate(pairs)) do entry
  id,(m,k)=entry;p=makecase(id,m,k);fixture=collect(reinterpret(Float32,read(joinpath(ROOT,@sprintf("study_case%02d_f32.bin",id)))));oracle=collect(reinterpret(Float32,read(joinpath(ROOT,@sprintf("oracle_case%02d_f32.bin",id)))))
  sizes=(p.ni,p.no,p.nw,p.nb);bases=map(n->CUDA.fill(-8192f0,n+32),sizes);arrays=map((b,n)->view(b,17:16+n),bases,sizes);input,output,weight,bias=arrays
  copyto!(input,fixture[1:p.ni]);copyto!(weight,fixture[p.ni+1:p.ni+p.nw]);copyto!(bias,fixture[p.ni+p.nw+1:end])
  args=Int32.((p.no,p.oc,p.m,p.w,p.h,p.ow,p.oh,p.k,p.k,1,1,1,1,1,1));run=factory(input,output,weight,bias,args)
  (p,arrays,bases,run,oracle)
 end
 for z in prepared;z[4]();end;CUDA.device_synchronize(blocking=true);allok=true
 for trial in 1:5
  prefix=Float32[];outputs=Float32[];guards=Float32[];times=Float64[];ok=true;err=0.
  for (p,arrays,bases,run,oracle) in prepared
   input,output,weight,bias=arrays;fill!(bases[2],-8192f0);run();CUDA.device_synchronize(blocking=true)
   start=time_ns();for _ in 1:10;run();end;CUDA.device_synchronize(blocking=true);push!(times,(time_ns()-start)*1e-9/10)
   append!(prefix,Array(input),Array(weight),Array(bias));append!(prefix,Float32[p.m,p.n,p.c,p.h,p.w,p.k,p.oh,p.ow,1,1,1,1,1,1,1,p.no,10,p.k in (1,3) ? p.k : 0])
   actual=Array(output);e=maximum(abs.(Float64.(actual).-Float64.(oracle)));err=max(err,e);ok&=all(isfinite,actual)&&actual==oracle;append!(outputs,actual)
   for base in bases;raw=Array(base);g=vcat(raw[1:16],raw[end-15:end]);ok&=all(==(-8192f0),g);append!(guards,g);end
  end
  allok&=ok;artifact=vcat(prefix,outputs,guards);path=joinpath(ENV["STUDY_OUTPUT_DIR"],"trial$(trial).bin");write(path,artifact)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":27756,\"max_abs_error\":%.12g,\"max_rel_error\":%s,\"timing_region\":\"sum_all12_source_depthwise_multiplier_kernel_cases_launch_loop_means\",\"jit_excluded\":true,\"internal_repeats\":10,\"timed_cases\":12,\"component_time_seconds\":[",trial,sum(times),ok ? "true" : "false",err,ok ? "0" : "null")
  for i in eachindex(times);i>1&&print(",");@printf("%.12g",times[i]);end
  @printf("],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",path,length(artifact))
 end
 allok||error("Full12caseconvolution/oracle/guards mismatch");println("PASS")
end
