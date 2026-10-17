using CUDA,Printf
const STUDY_CUDART="/usr/lib/x86_64-linux-gnu/libcudart.so.12.0.146"
const GUARD=-123.25f0
const STEPS=3
struct RuntimeExtent
 width::Csize_t
 height::Csize_t
 depth::Csize_t
end
struct RuntimePitched
 ptr::Ptr{Cvoid}
 pitch::Csize_t
 xsize::Csize_t
 ysize::Csize_t
end
runtime_check(code)=code==0 ? nothing : error("cudaRuntime status $code")
function fixture(w,h,d,dim3)
 CUDA.context();raw=Ref{Ptr{Cvoid}}(C_NULL);pitch=Ref{Csize_t}(0)
 if dim3
  z=Ref(RuntimePitched(C_NULL,0,0,0));runtime_check(ccall((:cudaMalloc3D,STUDY_CUDART),Cint,(Ref{RuntimePitched},RuntimeExtent),z,RuntimeExtent(4w,h,d)));raw[]=z[].ptr;pitch[]=z[].pitch
  z[].ysize==h||error("unexpected runtime3Dysize")
 else
  runtime_check(ccall((:cudaMallocPitch,STUDY_CUDART),Cint,(Ref{Ptr{Cvoid}},Ref{Csize_t},Csize_t,Csize_t),raw,pitch,4w,h))
 end
 stride=Int(pitch[]÷4);lengthphysical=stride*h*d;logical=Float32[(i%11==0 ? -80f0 : i%11==1 ? 80f0 : i%11==2 ? -0f0 : Float32((i*37)%513-256)/16f0) for i in 0:w*h*d-1];initial=fill(GUARD,lengthphysical);valid=falses(lengthphysical);goldlogical=Float64.(logical)
 for _ in 1:STEPS;goldlogical=1 ./ (1 .+ exp.(-goldlogical));end
 gold=fill(Float64(GUARD),lengthphysical)
 for z in 0:d-1,y in 0:h-1,x in 0:w-1
  li=(z*h+y)*w+x+1;pi=(z*h+y)*stride+x+1;initial[pi]=logical[li];gold[pi]=goldlogical[li];valid[pi]=true
 end
 pitched=unsafe_wrap(CuArray,CUDA.CuPtr{Float32}(raw[]),(lengthphysical,);own=false);simple=CuArray{Float32}(undef,length(logical))
 (w,h,d,dim3,stride,lengthphysical,logical,initial,goldlogical,gold,valid,pitched,simple,raw[])
end
function study_main(launch!)
 rep=isempty(ARGS) ? 5 : parse(Int,ARGS[1]);fs=[fixture(227,37,1,false),fixture(256,33,1,false),fixture(35,19,5,true),fixture(64,17,3,true)];prefix=vcat([vcat(Float32[f[1],f[2],f[3],f[4],STEPS,f[5],f[6],GUARD],f[7]) for f in fs]...)
 version=Ref{Cint}();runtime_check(ccall((:cudaRuntimeGetVersion,STUDY_CUDART),Cint,(Ref{Cint},),version));println("ACTUAL_CUDART version=",version[]," path=",STUDY_CUDART)
 for f in fs,mode in 1:2
  copyto!(mode==1 ? f[12] : f[13],mode==1 ? f[8] : f[7]);launch!(mode==1 ? f[12] : f[13],f,mode==1)
 end;CUDA.synchronize()
 for trial in 1:5
  times=zeros(Float64,8);gct=0.0;gcp=0
  for _ in 1:rep,(i,f) in enumerate(fs),mode in 1:2
   array=mode==1 ? f[12] : f[13];copyto!(array,mode==1 ? f[8] : f[7]);CUDA.synchronize();gb=Base.gc_num();st=time_ns()
   for _ in 1:STEPS;launch!(array,f,mode==1);end
   CUDA.synchronize();times[2(i-1)+mode]+=(time_ns()-st)*1e-9/rep;ga=Base.gc_num();gct+=(ga.total_time-gb.total_time)*1e-9;gcp+=ga.pause-gb.pause
  end
  got=Float32[];ok=true;ma=0.0;mr=0.0;padding=0
  for f in fs,mode in 1:2
   a=Array(mode==1 ? f[12] : f[13]);gold=mode==1 ? f[10] : f[9];valid=mode==1 ? f[11] : trues(length(a))
   for i in eachindex(a)
    if valid[i]
     err=abs(Float64(a[i])-gold[i]);ma=max(ma,err);mr=max(mr,err/max(abs(gold[i]),1e-30));ok&=isfinite(a[i])&&err<=2e-6+2e-6*abs(gold[i])
    else
     padding+=1;ok&=a[i]==GUARD
    end
   end
   append!(got,a)
  end
  art=vcat(prefix,got);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"padding_guard_count\":%d,\"timing_region\":\"sum8paths_3sigmoidsteps_kernelonly_resetexcluded\",\"jit_excluded\":true,\"internal_repeats\":%d,\"per_path_times_seconds\":[%s],\"gc_time_seconds_whole_region\":%.17g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",length(got),ma,mr,padding,rep,join(times,","),gct,gcp,path,length(art));ok||error("stableFloat64sigmoid/fullphysicalpaddingoracle")
 end
 for f in fs;CUDA.unsafe_free!(f[13]);runtime_check(ccall((:cudaFree,STUDY_CUDART),Cint,(Ptr{Cvoid},),f[14]));end
end
