include(joinpath(@__DIR__,"..","study_reference_kernels.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function run(top,sorted,experts,total,mapping,E,N,cum,maxpad,maxblocks,mapped)
 if N<Int32(1024)&&E<=Int32(64)
  stride=max(E,Int32(32));shared=((stride+1)*E+E+1)*4;@cuda threads=256+stride blocks=1 shmem=shared small_cuda!(top,sorted,experts,total,mapping,E,N,maxpad,maxblocks,stride,mapped)
 else
  padE=cld(E,Int32(32))*Int32(32);@cuda threads=1024 blocks=2 shmem=padE*4 general_cuda!(top,sorted,experts,total,mapping,E,padE,N,cum,maxpad,maxblocks,mapped)
  blocksY=cld(N,Int32(256));@cuda threads=256 blocks=(1,Int(blocksY)) scatter_cuda!(top,sorted,cum,mapping,E,N,blocksY,mapped)
 end
 return
end
study(run)
