const NX=32;const NY=24;const NZ=20;const SIZE=NX*NY*NZ
const COEFFICIENTS=Float32[-1/560,8/315,-.2,1.6,-1435/504]
@inline function laplace_native(field,i,a)
 @inbounds total=a[1]*field[i]
 for radius in Int32(1):Int32(4)
  @inbounds neighbors=field[i+radius]+field[i-radius]+field[i+radius*Int32(NX)]+field[i-radius*Int32(NX)]+field[i+radius*Int32(NX*NY)]+field[i-radius*Int32(NX*NY)]
  @inbounds total+=a[radius+1]*neighbors
 end
 return total
end
@inline function body(x,y,z,vsq,current_s,formal_current_r,formal_next_s,next_r,image,a)
 if Int32(4)<=x<Int32(NX-4)&&Int32(4)<=y<Int32(NY-4)&&Int32(4)<=z<Int32(NZ-4)
  i=x+y*Int32(NX)+z*Int32(NX*NY)+Int32(1)
  ds=laplace_native(current_s,i,a);dr=laplace_native(formal_current_r,i,a)
  @inbounds formal_next_s[i]=2f0*current_s[i]-formal_next_s[i]+vsq[i]*ds
  @inbounds next_r[i]=2f0*formal_current_r[i]-next_r[i]+vsq[i]*dr
  @inbounds image[i]=formal_next_s[i]*next_r[i]
 end
 return
end
function fixture()
 vsq=Float32[Float32(1+mod(i,3))/256f0 for i in 0:SIZE-1];cs=Float32[Float32(mod(i,17)-8)/16f0 for i in 0:SIZE-1];ns=Float32[Float32(mod(i,13)-6)/16f0 for i in 0:SIZE-1];cr=Float32[Float32(mod(i,11)-5)/32f0 for i in 0:SIZE-1];nr=Float32[Float32(mod(i,7)-3)/32f0 for i in 0:SIZE-1];return vsq,cs,ns,cr,nr
end
function study(run!)
 vsq,cs,ns,cr,nr=fixture();wantcr=Float64.(cr);wantnr=Float64.(nr);wantimage=fill(.5,SIZE)
 # Preserveactualnative mainbinding: formal_current_r=ns,formal_next_s=cr.
 for repeat in 1:5,z in 4:NZ-5,y in 4:NY-5,x in 4:NX-5
  i=x+y*NX+z*NX*NY+1
  function lap(field)
   result=Float64(COEFFICIENTS[1])*field[i]
   for radius in 1:4
    offsets=(radius,-radius,radius*NX,-radius*NX,radius*NX*NY,-radius*NX*NY);result+=Float64(COEFFICIENTS[radius+1])*sum(Float64(field[i+off]) for off in offsets)
   end
   return result
  end
  wantcr[i]=2*Float64(cs[i])-wantcr[i]+Float64(vsq[i])*lap(cs);wantnr[i]=2*Float64(ns[i])-wantnr[i]+Float64(vsq[i])*lap(ns);wantimage[i]=wantcr[i]*wantnr[i]
 end
 want=vcat(wantcr,wantnr,wantimage);V=CuArray(vsq);CS=CuArray(cs);NS=CuArray(ns);CR=CuArray(cr);NR=CuArray(nr);I=CuArray(fill(.5f0,SIZE));A=CuArray(COEFFICIENTS);run!(V,CS,NS,CR,NR,I,A);CUDA.synchronize()
 for trial in 1:5
  copyto!(CR,cr);copyto!(NR,nr);fill!(I,.5f0);CUDA.synchronize();start=time_ns();for repeat in 1:5;run!(V,CS,NS,CR,NR,I,A);end;CUDA.synchronize();elapsed=(time_ns()-start)*1e-9/5;got=vcat(Array(CR),Array(NR),Array(I));err=maximum(abs.(Float64.(got).-want));ok=all(isfinite,got)&&err<=.1;path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,got),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":46080,\"max_abs_error\":",err,",\"timing_region\":\"native_behavior_five_odd_recurrence_kernel_mean\",\"jit_excluded\":true,\"internal_repeats\":5,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float32\",\"count\":46080,\"byte_order\":\"little\"}}")
  ok||error("RTMnativebehavior independentrecurrence failed")
 end
end
