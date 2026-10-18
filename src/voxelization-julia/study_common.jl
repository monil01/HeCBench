const VG=64
const VGUARD=UInt32(0xdeadbeef)
function readwords(name)
 p=joinpath(@__DIR__,name);filesize(p)%4==0||error("word size")
 open(io->read!(io,Vector{UInt32}(undef,filesize(p)÷4)),p)
end
function study(allocate,build_scatter,extract,model)
 data=readwords("fixtures_u32.bin");idx=2;data[1]==4||error("case count");cases=Any[]
 for ci in 1:4
  n=Int(data[idx]);idx+=1;points=data[idx:idx+5n-1];idx+=5n
  sizes=(5n,1200000,160000,8000000,640000,1,4000000)
  all=map(sizes)do count
   d=allocate(count+2VG);copyto!(d,fill(VGUARD,count+2VG));d
  end
  views=map((d,count)->view(d,VG+1:VG+count),all,sizes)
  p,h,count,temp,index,real,feat=views;copyto!(p,points)
  c=(n=n,points=reinterpret(Float32,p),hash=h,count=count,temp=reinterpret(Float32,temp),indices=index,real=real,features=reinterpret(Float16,feat),all=all)
  push!(cases,c)
 end
 idx==length(data)+1||error("fixture trailingdata")
 function reset(c)
  fill!(c.hash,typemax(UInt32));fill!(c.count,UInt32(0));fill!(reinterpret(UInt32,c.temp),typemax(UInt32));fill!(c.indices,UInt32(0));fill!(c.real,UInt32(0));fill!(reinterpret(UInt32,c.features),UInt32(0));CUDA.device_synchronize(blocking=true)
 end
 for c in cases
  reset(c);build_scatter(c);CUDA.device_synchronize(blocking=true);real=Int(Array(c.real)[1]);real>0&&real%256==0||error("sourcefeaturefullgridcontrol");extract(c,real);CUDA.device_synchronize(blocking=true)
 end
 for trial in 1:5
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");rawpath=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"raw_trial_$(trial).bin");times=Float64[]
  open(rawpath,"w")do io
   write(io,UInt32(4))
   for c in cases
    reset(c);start=time_ns();build_scatter(c);CUDA.device_synchronize(blocking=true);push!(times,(time_ns()-start)*1e-9)
    real=Int(Array(c.real)[1]);real>0&&real%256==0||error("sourcefeaturefullgridcontrol");write(io,UInt32(c.n))
    for k in 1:6;write(io,Array(c.all[k]));end
    start=time_ns();extract(c,real);CUDA.device_synchronize(blocking=true);push!(times,(time_ns()-start)*1e-9)
    for k in (3,4,7);write(io,Array(c.all[k]));end
   end
  end
  run(`python3 $(joinpath(@__DIR__,"study_capture.py")) $rawpath $path $model $trial`)
  rawindex=strip(read(path*".index",String));filesize(path)==113497*4||error("canonical size")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",sum(times),",\"correct\":true,\"checked_values\":113497,\"max_abs_error\":0,\"max_rel_error\":null,\"timing_region\":\"sum4_completed_hash_scatter_plus_completed_feature_kernels\",\"jit_excluded\":true,\"internal_repeats\":1,\"timed_kernel_calls\":12,\"raw_output_identity_claim\":false,\"raw_state_index\":\"",rawindex,"\",\"stage_times_seconds\":[",join(times,","),"],\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"uint32\",\"count\":113497,\"byte_order\":\"little\"}}")
 end
end
