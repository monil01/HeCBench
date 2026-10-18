const BASE=dirname(@__FILE__);const N=1025;const G=16;const REPEAT=20;const SENTINEL=UInt32(0x5a5a5a5a)
function readwords(name,count)
 b=read(joinpath(BASE,name));length(b)==4*count||error("Missing/short/unexpectedfixture $name");collect(reinterpret(UInt32,b))
end
function snake_study(factory)
 a=readwords("study_reads_u32.bin",8*N);b=readwords("study_refs_u32.bin",8*N);oracle=readwords("oracle_results_i32.bin",26*N)
 A=CuArray(vcat(a,fill(SENTINEL,G)));B=CuArray(vcat(b,fill(SENTINEL,G)));O=CuArray(fill(reinterpret(Int32,SENTINEL),N+G));launch=factory(A,B,O,N)
 for t=0:25;launch(t);end;CUDA.device_synchronize(blocking=true)
 for trial=1:5
  prefix=vcat(Array(A)[1:8*N],Array(B)[1:8*N],UInt32.([N,100,25,26,REPEAT,256,8,0]));values=UInt32[];guards=UInt32[];times=Float64[];ok=true;mismatches=0
  for t=0:25
   fill!(O,reinterpret(Int32,SENTINEL));CUDA.device_synchronize(blocking=true);start=time_ns()
   for rep=1:REPEAT;launch(t);end
   CUDA.device_synchronize(blocking=true);push!(times,(time_ns()-start)*1e-9/REPEAT)
   out=reinterpret(UInt32,Array(O));expected=oracle[t*N+1:(t+1)*N];mismatches+=sum(out[1:N].!=expected);ok &= out[1:N]==expected&&all(<=(1),out[1:N]);append!(values,out[1:N]);append!(guards,out[N+1:end]);ok &= all(==(SENTINEL),out[N+1:end])
  end
  for x in(A,B);g=Array(x)[8*N+1:end];append!(guards,g);ok &= all(==(SENTINEL),g);end
  artifact=vcat(prefix,values,guards);path=joinpath(ENV["STUDY_OUTPUT_DIR"],"trial$(trial).bin");write(path,reinterpret(UInt8,artifact))
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%d,\"max_rel_error\":null,\"error_metric_scope\":\"Exactbinaryfilteroutputsandguardmismatchflag;relativeerrornotapplicable\",\"mismatch_output_values\":%d,\"timing_region\":\"sum26thresholds_meanR20_source_filter_kernel_launch_device_complete\",\"jit_excluded\":true,\"internal_repeats\":20,\"component_time_seconds\":[%s],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",length(values)+length(guards),mismatches>0 ? 1 : 0,mismatches,join(times,","),path,length(artifact));flush(stdout);ok||error("FullDNAfilterverdict/guardfailed")
 end
end
