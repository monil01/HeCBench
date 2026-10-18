using CUDA,Printf
const ROOT=abspath(joinpath(@__DIR__))
const G=-777f0
function study_main(model)
 fixture=reinterpret(Float64,read(joinpath(ROOT,"fixture.bin")));oracle=reinterpret(Float64,read(joinpath(ROOT,"oracle.bin")))
 cases=[];pos=1
 for ci in 1:3
  L=Int(fixture[pos]);n=Int(fixture[pos+1]);pos+=2
  arrays=Any[];whole=Any[]
  for (len,g,ty,guard) in [(n*6,6,Float32,G),(n*72,18,Float32,G),(n*72,18,Float32,G),(n*4,4,UInt64,999999),(n*4,4,UInt64,999999),(n*4,4,UInt64,999999),(n*4,4,UInt64,999999)]
   raw=ty.(fixture[pos:pos+len-1]);pos+=len;w=CuArray(vcat(fill(ty(guard),g),raw,fill(ty(guard),g)));push!(whole,w);push!(arrays,@view(w[g+1:g+len]))
  end
  outs=[CUDA.fill(G,n*6+12),CUDA.fill(G,n*72+36),CUDA.fill(G,n*72+36)]
  dst=@view outs[1][7:end-6];fb=@view outs[2][19:end-18];lb=@view outs[3][19:end-18]
  s,f,l,fw,b,fw3,b3=arrays;e=Int32(n÷2)
  launch_back(model,fb,lb,f,l,b,b3,e);launch_dslash(model,dst,s,f,l,fb,lb,fw,b,fw3,b3,e);CUDA.synchronize()
  push!(cases,(L,n,arrays,whole,outs,dst,fb,lb,e))
 end
 for trial in 1:5
  prefix=Float64[];got=Float64[];elapsed=0.0;path_times=Float64[]
  for (L,n,arrays,whole,outs,dst,fb,lb,e) in cases
   s,f,l,fw,b,fw3,b3=arrays
   for x in outs;fill!(x,G);end
   launch_back(model,fb,lb,f,l,b,b3,e);CUDA.synchronize()
   t=time_ns()
   for rep in 1:100
    launch_dslash(model,dst,s,f,l,fb,lb,fw,b,fw3,b3,e);CUDA.synchronize()
   end
   dt=(time_ns()-t)*1e-9/100;elapsed+=dt;push!(path_times,dt)
   append!(prefix,[L,n]);for a in arrays;append!(prefix,Float64.(Array(a)));end
   for a in outs;append!(got,Float64.(Array(a)));end
   for a in whole;append!(got,Float64.(Array(a)));end
  end
  ok=prefix==fixture;ma=0.0;mr=0.0
  for (a,b) in zip(got,oracle)
   er=abs(a-b);ma=max(ma,er);mr=max(mr,er/max(abs(b),1e-30));ok &= isfinite(a)&&er<=2e-5+2e-6*abs(b)
   if abs(b)>=700;ok &= a==b;end
  end
  out=vcat(prefix,got);file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(file,out)
  times=join(string.(path_times),",")
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"timing_region\":\"host_inclusive_sum3_dslash_100sync_launch_batches_div100_backward_setup_excluded\",\"jit_excluded\":true,\"internal_repeats\":100,\"path_times_seconds\":[%s],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,string(ok),length(out),ma,mr,times,file,length(out))
 end
end
