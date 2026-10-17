include(joinpath(@__DIR__,"study_kernels.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
const BACKEND=CUDA.CUDABackend()
const GENERAL=general_ka!(BACKEND,1024)
const SCATTER=scatter_ka!(BACKEND,(256,1))
function run(top,sorted,experts,total,mapping,E,N,cum,maxpad,maxblocks,mapped)
 if N<Int32(1024)&&E<=Int32(64)
  stride=max(E,Int32(32));words=Int((stride+Int32(1))*E+E+Int32(1));threads=Int(Int32(256)+stride);small_ka!(BACKEND,threads)(top,sorted,experts,total,mapping,E,N,maxpad,maxblocks,stride,mapped,Val(words);ndrange=threads)
 else
  padE=cld(E,Int32(32))*Int32(32);GENERAL(top,sorted,experts,total,mapping,E,padE,N,cum,maxpad,maxblocks,mapped,Val(Int(padE));ndrange=2048)
  blocksY=cld(N,Int32(256));SCATTER(top,sorted,cum,mapping,E,N,blocksY,mapped;ndrange=(256,Int(blocksY)))
 end
 return
end
study(run)
