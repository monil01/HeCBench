# Adapted from repository cmembench/study_constant.jl: actual LLVM AS4 constant symbol.
using CUDA
@inline function load_k_component(index::Int32)
 Base.llvmcall(("""
 @fhd_study_k = weak addrspace(4) externally_initialized global [12288 x float] zeroinitializer, align 16
 define float @entry(i32 %idx) #0 {
  %wide = sext i32 %idx to i64
  %p = getelementptr inbounds [12288 x float], [12288 x float] addrspace(4)* @fhd_study_k, i64 0, i64 %wide
  %v = load float, float addrspace(4)* %p, align 4
  ret float %v
 }
 attributes #0 = { alwaysinline }
 ""","entry"),Float32,Tuple{Int32},index)
end
function upload_constant(symbol,host,offset,count)
 GC.@preserve host CUDA.cuMemcpyHtoD_v2(symbol,pointer(host,offset),Csize_t(count*4))
end
function read_constant(symbol)
 out=zeros(Float32,12288);GC.@preserve out CUDA.cuMemcpyDtoH_v2(pointer(out),symbol,Csize_t(49152));out
end
