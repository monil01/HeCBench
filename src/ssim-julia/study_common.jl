using Printf
const BASE=dirname(@__FILE__);const G=16;const SENT=UInt32(0x4b123456);const POISON=UInt32(0x7fc12345)
include("study_cases.jl")
function readtyped(::Type{T},name,n)where T
 b=read(joinpath(BASE,name));length(b)==sizeof(T)*n||error("Unexpectedfixture $name");collect(reinterpret(T,b))
end
function gpu_reduce!(s,t)
 s.launch(t)
 result=sum(view(s.O,1:t.count))
 result isa Float32||error("SourceusesFloat32GPUreductiontohost")
 result
end
function ssim_study(makearray,factory)
 input=readtyped(UInt32,"input_u32.bin",121856);oracle=readtyped(Float64,"oracle_f64.bin",68247);states=Any[]
 for c in CASES
  a=vcat(input[c.io+1:c.io+c.gcount],fill(SENT,G));b=vcat(input[c.io+c.gcount+1:c.io+2*c.gcount],fill(SENT,G));reset=fill(SENT,c.cap+G);A=makearray(collect(reinterpret(Float32,a)));B=makearray(collect(reinterpret(Float32,b)));O=makearray(collect(reinterpret(Float32,reset)));launch=factory(A,B,O);s=(case=c,a=a,b=b,reset=reset,A=A,B=B,O=O,launch=launch);CUDA.device_synchronize(blocking=true)
  for ti=c.tbegin+1:c.tend;gpu_reduce!(s,TILES[ti]);end;CUDA.device_synchronize(blocking=true);push!(states,s)
 end
 for trial=1:5
  prefix=Float64[];values=Float64[];guards=Float64[];times=Float64[];ok=true;maxabs=0.;maxbudget=0.
  for(ci,s)in enumerate(states)
   c=s.case;append!(prefix,Float64.(reinterpret(UInt32,Array(s.A))[1:c.gcount]));append!(prefix,Float64.(reinterpret(UInt32,Array(s.B))[1:c.gcount]));params=UInt32.([c.dx,c.dy,c.dz,c.bx,c.by,c.bz,c.gcount,7,343,0,0,0,0,ci-1,3,0]);params[10]=reinterpret(UInt32,1f0);params[11]=reinterpret(UInt32,343f0/342f0);params[12]=reinterpret(UInt32,.01f0);params[13]=reinterpret(UInt32,.03f0);append!(prefix,Float64.(params));total=0f0;expectedTotal=0.
   for ti=c.tbegin+1:c.tend
    t=TILES[ti];append!(prefix,Float64.([t.bx,t.by,t.bz,t.gx,t.gy,t.gz,t.ox,t.oy,t.oz,t.count,ci-1,ti-c.tbegin-1]));reset=copy(s.reset);reset[1:t.count].=POISON;copyto!(s.O,collect(reinterpret(Float32,reset)));CUDA.device_synchronize(blocking=true);start=time_ns();reduced=gpu_reduce!(s,t);total+=reduced;CUDA.device_synchronize(blocking=true);push!(times,(time_ns()-start)*1e-9)
    raw=reinterpret(UInt32,Array(s.O));actual=Float64.(reinterpret(Float32,raw[1:t.count]));expected=oracle[t.oo+1:t.oo+t.count];expectedSum=sum(expected);expectedTotal+=expectedSum;extra=Float64[reduced,total];want=Float64[expectedSum,expectedTotal];allactual=vcat(actual,extra);allexpect=vcat(expected,want);diff=abs.(allactual.-allexpect);budget=2e-5 .+ 3e-5 .* abs.(allexpect);ok &= all(isfinite,allactual)&&all(diff.<=budget);maxabs=max(maxabs,maximum(diff));maxbudget=max(maxbudget,maximum(diff./budget));append!(values,allactual);tail=raw[t.count+1:end];append!(guards,Float64.(tail));ok &= all(==(SENT),tail)
   end
   normalized=total/Float32(c.voxels);push!(values,Float64(normalized));err=abs(Float64(normalized)-expectedTotal/c.voxels);budget=2e-5+3e-5*abs(expectedTotal/c.voxels);ok &= isfinite(normalized)&&err<=budget;maxabs=max(maxabs,err);maxbudget=max(maxbudget,err/budget)
   for(a,h)in ((s.A,s.a),(s.B,s.b));raw=reinterpret(UInt32,Array(a));ok &= raw==h;append!(guards,Float64.(raw[c.gcount+1:end]));end
  end
  art=vcat(prefix,values,guards);path=joinpath(ENV["STUDY_OUTPUT_DIR"],"trial$(trial).bin");write(path,reinterpret(UInt8,art))
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"max_normalized_budget_error\":%.17g,\"timing_region\":\"sum10tiles_compute_ssim7_GPUreduce_to_host_F32accum_device_complete\",\"jit_excluded\":true,\"internal_repeats\":1,\"source_tile_invocations\":10,\"component_time_seconds\":[%s],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",length(values)+length(guards),maxabs,maxbudget,join(times,","),path,length(art));flush(stdout);ok||error("FullSSIMfields/sums/normalizedmetrics/input/guardsfailed")
 end
end
