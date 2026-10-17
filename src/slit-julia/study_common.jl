using CUDA,Printf,SHA
const SN=32
function study_inputs()
 N=SN;arrays=[zeros(ComplexF64,N,N) for _ in 1:3]
 for y in 0:N-1,x in 0:N-1
  if 8<=abs(x-N÷2)<=10 && abs(y-N÷2)<=4;arrays[1][x+1,y+1]=1;end
  arrays[3][x+1,y+1]=complex(((7x+3y)%23-11)/32,((5x+8y)%19-9)/64)
 end
 arrays[2][2,3]=1
 arrays
end
function direct_dft(a)
 N=size(a,1);tw=[cis(-2pi*k*j/N) for k in 0:N-1,j in 0:N-1];tmp=zeros(ComplexF64,N,N);out=similar(tmp)
 for y in 1:N,k in 1:N
  z=0.0+0.0im;for x in 1:N;z+=a[x,y]*tw[k,x];end;tmp[k,y]=z
 end
 for kx in 1:N,ky in 1:N
  z=0.0+0.0im;for y in 1:N;z+=tmp[kx,y]*tw[ky,y];end;out[kx,ky]=z
 end
 out
end
function study_main(allocate!)
 repeat=isempty(ARGS) ? 100 : parse(Int,ARGS[1]);N=SN;inputs=study_inputs();gold=direct_dft.(inputs);ds=[allocate!(a) for a in inputs];handles=[Ref{CUDA.CUFFT.cufftHandle}() for _ in inputs]
 println("STUDY_VENDOR {\"library\":\"",realpath(CUDA.CUFFT.libcufft),"\",\"sha256\":\"",bytes2hex(sha256(read(realpath(CUDA.CUFFT.libcufft)))),"\",\"version\":\"",CUDA.CUFFT.version(),"\",\"plan_API\":\"cufftPlan2d_Z2Z\",\"stream\":0}")
 for f in 1:3
  CUDA.CUFFT.cufftPlan2d(handles[f],N,N,CUDA.CUFFT.CUFFT_Z2Z);copyto!(ds[f],inputs[f]);CUDA.CUDACore.device_synchronize();CUDA.CUFFT.cufftExecZ2Z(handles[f][],ds[f],ds[f],CUDA.CUFFT.CUFFT_FORWARD);CUDA.CUDACore.device_synchronize()
 end
 prefix=vcat([collect(reinterpret(Float64,vec(a))) for a in inputs]...)
 for trial in 1:5
  elapsed=0.0
  for f in 1:3,_ in 1:repeat
   copyto!(ds[f],inputs[f]);CUDA.CUDACore.device_synchronize();start=time_ns()
   CUDA.CUFFT.cufftExecZ2Z(handles[f][],ds[f],ds[f],CUDA.CUFFT.CUFFT_FORWARD);CUDA.CUDACore.device_synchronize();elapsed+=(time_ns()-start)*1e-9/repeat
  end
  got=Array.(ds);vals=Float64[];err=0.0;ok=true
  for f in 1:3
   out=got[f];inten=abs2.(out);err=max(err,maximum(abs.(real.(out).-real.(gold[f]))),maximum(abs.(imag.(out).-imag.(gold[f]))),maximum(abs.(inten.-abs2.(gold[f]))));ok&=all(isfinite,inten)&&abs(sum(inten)-N*N*sum(abs2,inputs[f]))<1e-8&&abs(out[1,1]-sum(inputs[f]))<1e-10
   if f==2
    for ky in 0:N-1,kx in 0:N-1;ok&=abs(out[kx+1,ky+1]-cis(-2pi*(kx+2ky)/N))<1e-10;end
   end
   append!(vals,reinterpret(Float64,vec(out)));append!(vals,vec(inten))
  end
  ok&=err<1e-8;art=vcat(prefix,vals);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":0.0,\"timing_region\":\"sum_three_inplace_cufft_Z2Z_forward_exec_device_sync_only\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,ok ? "true" : "false",length(vals),err,repeat,path,length(art));ok||error("slit directDFT/phase/Parseval contract")
 end
 for h in handles;CUDA.CUFFT.cufftDestroy(h[]);end
end
