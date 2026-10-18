using CUDA,Printf
const ROOT=@__DIR__
function decode_values(v,ty)
 ty==1 ? Float64.(reinterpret(Float16,v)) : Float64.(v)
end
function encode_values(v,ty)
 ty==1 ? collect(reinterpret(UInt32,Float16.(v))) : ty==2 ? Float32.(v) : Float64.(v)
end
function study_main(model)
 fixture=collect(reinterpret(Float64,read(joinpath(ROOT,"fixture.bin"))));gold=collect(reinterpret(Float64,read(joinpath(ROOT,"oracle.bin"))))
 cases=[];pos=1
 for ci in 1:12
  ty,n,t,b,h=Int.(fixture[pos:pos+4]);pos+=5;lanes=ty==1 ? 2 : 1
  a=fixture[pos:pos+n*lanes-1];pos+=n*lanes;raw=encode_values(vcat(fill(-777.0,8*lanes),a,fill(-777.0,8*lanes)),ty)
  w=CuArray(raw);payload=@view w[9:end-8];initial=copy(raw)
  launch_ert(model,payload,n,t,b,h);CUDA.synchronize();push!(cases,(ty,n,t,b,h,lanes,w,payload,initial))
 end
 for trial in 1:5
  prefix=Float64[];out=Float64[];echo=Float64[];elapsed=0.0;times=Float64[]
  for (ty,n,t,b,h,lanes,w,payload,initial) in cases
   copyto!(w,initial);CUDA.synchronize()
   actual=decode_values(Array(w),ty);append!(prefix,[ty,n,t,b,h]);append!(prefix,actual[8*lanes+1:end-8*lanes])
   stamp=time_ns();launch_ert(model,payload,n,t,b,h);CUDA.synchronize();dt=(time_ns()-stamp)*1e-9;elapsed+=dt;push!(times,dt)
   append!(out,decode_values(Array(w),ty));append!(echo,actual)
  end
  ok=reinterpret(UInt64,prefix)==reinterpret(UInt64,fixture) && out==gold
  values=vcat(prefix,out,echo);file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(file,values)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":0,\"max_rel_error\":0,\"path_times_seconds\":[%s],\"timing_region\":\"sum12_hostinclusive_freshstate_single_kernel_regions\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,string(ok),length(values),join(string.(times),","),file,length(values))
 end
end
