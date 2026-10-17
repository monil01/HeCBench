using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function fwt_study(factory)
 n=65536;k=128;repeats=1
 host=Float32[(mod(i*13+div(i,31),31)-15)/32 for i in 0:n-1]
 padded=zeros(Float32,n);padded[1:k]=Float32[(mod(j*7,17)-8)/16 for j in 0:k-1]
 expected=Float32[sum(Float64(host[xor(i,j)+1])*Float64(padded[j+1]) for j in 0:k-1) for i in 0:n-1]
 data=CuArray(host);kernel=CuArray(padded);operation! =factory(data,kernel,n)
 reset! =()->begin;copyto!(data,host);copyto!(kernel,padded);end
 readback=()->begin
  got=Array(data);l2=sqrt(sum((Float64(got[i])-Float64(expected[i]))^2 for i in eachindex(got))/sum(abs2,Float64.(expected)))
  println("STUDY_NATIVE_RELATIVE_L2 ",l2);isfinite(l2)&&l2<1e-6||error("original L2 criterion failed");got
 end
 println("STUDY_CONTRACT XOR_dyadic_convolution n=65536 kernel=128 repeats=1 full_output=65536 original_l2_lt=1e-6 inherited_Julia_radix2_global_vs_native_radix4_shared")
 OwnedStudy.run_trials!(operation!,reset!,readback,expected,repeats;atol=1e-5,rtol=0.)
end
