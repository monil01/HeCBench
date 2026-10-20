using CUDA
include(joinpath(@__DIR__,"study_source_atan.jl"))
function probe!(input,out,n)
 i=(blockIdx().x-1)*blockDim().x+threadIdx().x
 if i<=n;x=input[i];out[2*i-1]=source_atanf(x);out[2*i]=muladd(source_atanf(x),Float32(1f0/3.14159f0),.5f0);end
 nothing
end
f=open(ARGS[1]);n=Int(read(f,Int32));input=read!(f,zeros(Float32,n));close(f);di=CuArray(input);full=CuArray(reinterpret(Float32,fill(UInt8(165),(2*n+128)*4)));out=@view full[65:end-64]
CUDA.code_ptx(open(ARGS[2],"w"),probe!,Tuple{typeof(CUDA.cudaconvert(di)),typeof(CUDA.cudaconvert(out)),Int64};kernel=true)
for t=0:5
 CUDA.@cuda threads=256 blocks=cld(n,256) probe!(di,out,n);CUDA.device_synchronize();t==0&&continue
 host=Array(full);physical=collect(reinterpret(UInt8,host));guards=[physical[1:256];physical[end-255:end]];all(==(165),guards)||error("guard")
 name="math_trial$(t).bin";values=[Float64.(input);Float64.(host[65:end-64])];write(name,values);write("math_trial$(t).guards.u8",guards)
 println("STUDY_TRIAL {\"trial\":$t,\"correct\":true,\"time_seconds\":0,\"checked_values\":$(length(values)),\"max_abs_error\":null,\"max_rel_error\":null,\"timing_region\":\"diagnostic_only_unmeasured\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"$name\",\"dtype\":\"float64\",\"count\":$(length(values)),\"byte_order\":\"little\"}}")
end
