include(joinpath(@__DIR__,"reference_history","original_kernel_module.jl"))
x=zeros(Float32,513);x[1]=127;x[end]=-128;di=CuArray(x);out=CUDA.zeros(Int8,513);scale=CUDA.zeros(Float32,1)
@cuda threads=513 blocks=1 dynamic_scaled_int8_quant_kernel!(di,out,scale,Int32(513));CUDA.synchronize()
println("ORIGINAL_JULIA_SYMMETRIC_513 scale=",Array(scale)[1]," lastcode=",Array(out)[end]," expected_scale=",128f0/127f0)
for T in (Float16,CUDA.CUDACore.BFloat16,Float32)
 x=T.(zeros(Float32,1024));x[1]=T(127);x[end]=T(-128);di=CuArray(x);out=CUDA.zeros(Int8,1024);scale=CUDA.zeros(Float32,1)
 @cuda threads=1024 blocks=1 dynamic_scaled_int8_quant_kernel!(di,out,scale,Int32(1024));CUDA.synchronize()
 println("REAL_STORAGE_WITNESS type=",T," bytes_per_element=",sizeof(T)," scale=",Array(scale)[1]," lastcode=",Array(out)[end])
end
