# Adapted from cmembench/FHD: actual LLVM addrspace(4), not a global-memory LUT.
# Backend-specific extension is disclosed for CUDAjl, KA and JACC alike.
using CUDA
@inline function load_bound(index::Int32)
 Base.llvmcall(("""
 @tpacf_study_bins = weak addrspace(4) externally_initialized global [63 x double] zeroinitializer, align 16
 define double @entry(i32 %idx) #0 {
  %wide = sext i32 %idx to i64
  %p = getelementptr inbounds [63 x double], [63 x double] addrspace(4)* @tpacf_study_bins, i64 0, i64 %wide
  %v = load double, double addrspace(4)* %p, align 8
  ret double %v
 }
 attributes #0 = { alwaysinline }
 ""","entry"),Float64,Tuple{Int32},index)
end
function upload_bins(symbol,values)
 GC.@preserve values CUDA.CUDACore.cuMemcpyHtoD_v2(symbol,pointer(values),Csize_t(504))
 nothing
end
function read_bins(symbol)
 a=zeros(Float64,63);GC.@preserve a CUDA.CUDACore.cuMemcpyDtoH_v2(pointer(a),symbol,Csize_t(504));a
end
