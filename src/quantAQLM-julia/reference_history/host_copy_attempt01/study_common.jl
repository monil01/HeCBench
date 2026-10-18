
const BASE=dirname(@__FILE__);const G=16;const R=20
function quant_study(factory)
 sms=CUDA.attribute(CUDA.device(),CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
 cases=map(enumerate(((12288,4096),(96,512))))do (caseid,(m,k))
  nc=m*k÷8;ni=4k;nb=3*65536*8;no=4m;lens=(nc,ni,no,nb)
  input=collect(reinterpret(UInt16,read(joinpath(BASE,"study_case$(caseid)_u16.bin"))));oracle=collect(reinterpret(UInt16,read(joinpath(BASE,"oracle_case$(caseid)_u16.bin"))))
  A=CuArray(fill(UInt16(0x7b00),nc+G));B=CuArray(fill(reinterpret(Float16,UInt16(0x7b00)),ni+G));C=CuArray(fill(reinterpret(Float16,UInt16(0x7b00)),no+G));book=CuArray(fill(reinterpret(Float16,UInt16(0x7b00)),nb+G));arrays=(A,B,C,book)
  copyto!(A,1,input,1,nc);copyto!(B,1,reinterpret(Float16,input[nc+1:nc+ni]),1,ni);copyto!(book,1,reinterpret(Float16,input[nc+ni+1:end]),1,nb)
  waves=0;tm=17;while tm>16;waves+=1;tm=cld(m,waves*sms);end;threads=32tm;blocks=cld(m,tm);launch=factory(arrays,m,k,blocks,threads)
  for b=0:3;launch(b*k,b*m);end;CUDA.device_synchronize(blocking=true)
  (m,k,lens,arrays,oracle,launch)
 end
 for trial=1:5
  prefix=UInt32[];values=UInt32[];guards=UInt32[];times=Float64[];ok=true;mismatches=0
  for (caseid,(m,k,lens,arrays,oracle,launch))in enumerate(cases)
   elapsed=0.
   for b=0:3,repeat=1:R;CUDA.device_synchronize(blocking=true);start=time_ns();launch(b*k,b*m);CUDA.device_synchronize(blocking=true);elapsed+=(time_ns()-start)*1e-9;end
   push!(times,elapsed/R)
   for j=1:4
    host=Array(arrays[j]);bits=j==1 ? host : reinterpret(UInt16,host);n=lens[j];append!(guards,UInt32.(bits[n+1:end]))
    if j==3
     for i=1:n;push!(values,UInt32(bits[i]));pass=bits[i]==oracle[i]&&isfinite(host[i]);ok &= pass;mismatches+=!pass;end
    else
     for i=1:2:n;push!(prefix,UInt32(bits[i])|(UInt32(bits[i+1])<<16));end
    end
   end
   append!(prefix,UInt32.([m,k,4,65536,m÷3,2*(m÷3),m,k÷64,R,9,8,16,0,0,0,caseid]))
  end
  ok &= all(==(UInt32(0x7b00)),guards);out=vcat(prefix,values,guards);path=joinpath(ENV["STUDY_OUTPUT_DIR"],"trial$(trial).bin");write(path,reinterpret(UInt8,out))
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%d,\"max_rel_error\":null,\"error_metric_scope\":\"max_abs_error is halfbit mismatchflag; relativeerror uncomputed; separate denseFP64quality\",\"mismatch_half_values\":%d,\"timing_region\":\"sum2_fourvectors_meanR20_original_quant_matvec_complete\",\"jit_excluded\":true,\"internal_repeats\":20,\"component_time_seconds\":[%s],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",length(values)+length(guards),mismatches>0 ? 1 : 0,mismatches,join(times,","),path,length(out));flush(stdout);ok||error("Fullquantizedhalfbit/codebook/FMA/tree/guardsfailed")
 end
end
