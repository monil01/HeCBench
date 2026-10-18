using CUDA,Printf
function case_input(::Type{T},n) where T
 out=fill(T(0),n+64÷sizeof(T))
 fill!(reinterpret(UInt8,out),0xa5)
 for i in 0:n-1
  x=UInt64(i+1)*UInt64(0x9e3779b97f4a7c15);x ⊻= x>>30;x*=UInt64(0xbf58476d1ce4e5b9);x ⊻=x>>27
  out[32÷sizeof(T)+i+1]=x%T
 end
 out
end
function reverse_study(factory)
 vector=!isempty(ARGS)&&ARGS[1]=="1";shapes=vector ? [(32,64),(64,96)] : [(17,19),(33,35)]
 cases=[];repeats=7
 for (nr,nc) in shapes,T in (UInt8,UInt16,UInt32,UInt64),rm in (true,false),ar in (true,false)
  n=nr*nc;offset=32÷sizeof(T);host=case_input(T,n);expected=copy(host)
  for i in 0:n-1
   row=rm ? i÷nc : i%nr;col=rm ? i%nc : i÷nr
   sr=ar ? nr-row-1 : row;sc=ar ? col : nc-col-1;j=rm ? sr*nc+sc : sc*nr+sr
   expected[offset+i+1]=host[offset+j+1]
  end
  input=CuArray(host);out=CuArray(case_input(T,n));launch! = factory(input,out,Int32(nr),Int32(nc),rm,ar,Int32(offset))
  push!(cases,(input=input,out=out,host=host,expected=expected,launch=launch!))
 end
 function suite!()
  for c in cases,_ in 1:repeats;c.launch();end
  CUDA.device_synchronize(blocking=true)
 end
 suite!();allok=true
 for trial in 1:5
  for c in cases;fill!(reinterpret(UInt8,c.out),0xa5);end
  CUDA.device_synchronize(blocking=true);t=time_ns();suite!();secs=(time_ns()-t)*1e-9/repeats
  artifact=UInt8[];ok=true
  for c in cases
   back=Array(c.input);ok &=back==c.host;append!(artifact,reinterpret(UInt8,back))
  end
  for c in cases
   got=Array(c.out);ok &=got==c.expected;append!(artifact,reinterpret(UInt8,got))
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,artifact);allok &=ok
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%s,\"max_rel_error\":null,\"timing_region\":\"32case_reverse_kernels_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint8\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,secs,ok ? "true" : "false",length(artifact),ok ? "0" : "null",repeats,path,length(artifact))
 end
 allok||error("exacttypedfullmatrix/input/guardcheck failed");println("PASS")
end
