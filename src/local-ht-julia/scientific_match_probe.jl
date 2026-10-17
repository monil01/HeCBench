using CUDA
@inline match_any64(mask::UInt32,value::UInt64)=ccall("llvm.nvvm.match.any.sync.i64",llvmcall,UInt32,(UInt32,UInt64),mask,value)
function probe!(out)
 lane=threadIdx().x-Int32(1)
 out[lane+Int32(1)]=match_any64(UInt32(0xffffffff),UInt64(mod(lane,Int32(4))))
 return
end
out=CUDA.zeros(UInt32,32)
@cuda threads=32 blocks=1 probe!(out)
CUDA.synchronize()
got=Array(out);expected=UInt32[UInt32(0x11111111)<<mod(i,4) for i in 0:31]
println("MATCH_ANY64_GOT=",got)
println("MATCH_ANY64_EXPECTED=",expected)
got==expected||error("Warpcohortmatchmismatch")
println("MATCH_ANY64_PASS")
