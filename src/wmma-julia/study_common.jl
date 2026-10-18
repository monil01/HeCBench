using Printf
include("study_cases.jl")
const BASE=@__DIR__;const G=32;const SENT=UInt32(0x4b123456);const HSENT=UInt16(0x3555)
function wmma_study(makearray,factory)
 input=collect(reinterpret(UInt32,read(joinpath(BASE,"input_u32.bin"))));oracle=collect(reinterpret(Float32,read(joinpath(BASE,"oracle_f32.bin"))));states=Any[]
 for c in CASES
  na=c.m*c.k;nb=c.n*c.k;nc=c.m*c.n;A=makearray(vcat(collect(reinterpret(Float16,UInt16.(input[c.io+1:c.io+na]))),fill(reinterpret(Float16,HSENT),G)));B=makearray(vcat(collect(reinterpret(Float16,UInt16.(input[c.io+na+1:c.io+na+nb]))),fill(reinterpret(Float16,HSENT),G)));C=makearray(vcat(collect(reinterpret(Float32,input[c.io+na+nb+1:c.io+na+nb+nc])),fill(reinterpret(Float32,SENT),G)));D=makearray(fill(reinterpret(Float32,SENT),nc+G));launch=factory(A,B,C,D,c)
  for j=1:30;launch();end;CUDA.device_synchronize(blocking=true);push!(states,(c=c,A=A,B=B,C=C,D=D,launch=launch))
 end
 for trial=1:5
  prefix=UInt32[];outputs=UInt32[];guards=UInt32[];times=Float64[];ok=true;mismatches=0
  for s in states
   c=s.c;na=c.m*c.k;nb=c.n*c.k;nc=c.m*c.n;A=Array(s.A);B=Array(s.B);C=Array(s.C);append!(prefix,UInt32.(reinterpret(UInt16,A[1:na])));append!(prefix,UInt32.(reinterpret(UInt16,B[1:nb])));append!(prefix,reinterpret(UInt32,C[1:nc]));append!(prefix,UInt32.([c.impl,c.m,c.n,c.k,c.repeats,c.impl==0 ? 32 : 128,c.impl==0 ? 1 : 4,16,16,16]));wanted=input[c.io+1:c.io+na+nb+nc];ok &=vcat(UInt32.(reinterpret(UInt16,A[1:na])),UInt32.(reinterpret(UInt16,B[1:nb])),reinterpret(UInt32,C[1:nc]))==wanted
   reset=fill(SENT,nc+G);reset[1:nc].=UInt32(0x7fc12345);copyto!(s.D,collect(reinterpret(Float32,reset)));CUDA.device_synchronize(blocking=true);start=time_ns();for rep=1:c.repeats;s.launch();end;CUDA.device_synchronize(blocking=true);push!(times,(time_ns()-start)*1e-9)
   D=Array(s.D);actual=reinterpret(UInt32,D[1:nc]);expected=reinterpret(UInt32,oracle[c.oo+1:c.oo+nc]);mismatches+=count(actual.!=expected);ok &=all(isfinite,D)&&actual==expected;append!(outputs,actual)
   for(a,h,sentinel)in ((s.A,na,UInt32(HSENT)),(s.B,nb,UInt32(HSENT)),(s.C,nc,SENT),(s.D,nc,SENT))
    host=Array(a);bits=eltype(host)==Float16 ? UInt32.(reinterpret(UInt16,host)) : collect(reinterpret(UInt32,host));tail=bits[h+1:end];ok &=all(==(sentinel),tail);append!(guards,tail)
   end
   ok &=UInt32.(reinterpret(UInt16,Array(s.A)[1:na]))==input[c.io+1:c.io+na]&&UInt32.(reinterpret(UInt16,Array(s.B)[1:nb]))==input[c.io+na+1:c.io+na+nb]&&reinterpret(UInt32,Array(s.C)[1:nc])==input[c.io+na+nb+1:c.io+na+nb+nc]
  end
  art=vcat(prefix,outputs,guards);path=joinpath(ENV["STUDY_OUTPUT_DIR"],"trial$(trial).bin");write(path,reinterpret(UInt8,art))
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":null,\"max_rel_error\":null,\"exact_raw_bit_mismatches\":%d,\"timing_region\":\"sum11cases_R100_sourceWMMA16fragments_FP16_F32_devicecomplete\",\"jit_excluded\":true,\"internal_repeats\":100,\"case_count\":11,\"component_time_seconds\":[%s],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",length(outputs)+length(guards),mismatches,join(times,","),path,length(art));flush(stdout);ok||error("TrueWMMAfulloutputs/inputs/guardsfailed")
 end
end
