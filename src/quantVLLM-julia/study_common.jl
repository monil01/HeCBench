using CUDA,Printf
const ST_INPUT=collect(reinterpret(UInt32,read(joinpath(@__DIR__,"fixture_bits.bin"))))
const ST_GOLD=collect(reinterpret(Float64,read(joinpath(@__DIR__,"expected_full_outputs.bin"))))
function st_fixtures()
 fs=[];pos=1
 for _ in 1:6
  t,h,nt,scale_bits,azp=ST_INPUT[pos:pos+4];pos+=5;len=Int(h*nt);bits=ST_INPUT[pos:pos+len-1];pos+=len
  T=t==1 ? Float16 : t==2 ? CUDA.CUDACore.BFloat16 : Float32
  input=t==3 ? collect(reinterpret(Float32,bits)) : collect(reinterpret(T,UInt16.(bits)))
  push!(fs,(Int(t),Int(h),Int(nt),reinterpret(Float32,scale_bits),Int32(azp),input))
 end
 @assert pos==length(ST_INPUT)+1
 fs
end
function study_main(launch!)
 rep=isempty(ARGS) ? 20 : parse(Int,ARGS[1]);paths=[]
 for f in st_fixtures()
  di=CuArray(f[6]);n=f[2]*f[3]
  println("REAL_INPUT_STORAGE type=",eltype(di)," bytes_per_element=",sizeof(eltype(di))," hidden=",f[2]," tokens=",f[3])
  for mode in 1:4
   out=CUDA.fill(Int8(-99),n+17);sc=CUDA.fill(-12345f0,f[3]+17);zp=CUDA.fill(Int32(-123456),f[3]+17);push!(paths,(mode,f,di,out,sc,zp))
  end
 end
 for p in paths;launch!(p);end;CUDA.synchronize()
 for trial in 1:5
  got=Float64[];times=Float64[];gct=0.;gcp=0
  for p in paths
   mode,f,di,out,sc,zp=p;CUDA.fill!(out,Int8(-99));CUDA.fill!(sc,-12345f0);CUDA.fill!(zp,Int32(-123456));CUDA.synchronize();gb=Base.gc_num();st=time_ns()
   for _ in 1:rep;launch!(p);end
   CUDA.synchronize();push!(times,(time_ns()-st)*1e-9/rep);ga=Base.gc_num();gct+=(ga.total_time-gb.total_time)*1e-9;gcp+=ga.pause-gb.pause
   append!(got,Float64.(Array(out)));append!(got,Float64.(Array(sc)));append!(got,Float64.(Array(zp)))
  end
  ok=got==ST_GOLD;ma=maximum(abs.(got.-ST_GOLD));file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(file,vcat(Float64.(ST_INPUT),got))
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"timing_region\":\"sum24_FP16_BF16_FP32_static_dynamic_quantization_kernels_only\",\"jit_excluded\":true,\"internal_repeats\":%d,\"per_path_times_seconds\":[%s],\"gc_time_seconds_whole_region\":%.17g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",length(got),ma,rep,join(times,","),gct,gcp,file,length(ST_INPUT)+length(got))
 end
end
