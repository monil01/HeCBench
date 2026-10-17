using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function keogh_kernel!(subject,avgs,stds,out,lower,upper,M::Int32,N::Int32)
 cache=@cuDynamicSharedMem(Float32,Int(M+blockDim().x));lid=threadIdx().x-Int32(1);blocksize=blockDim().x*(blockIdx().x-Int32(1));idx=blocksize+lid
 k=lid
 while k<blockDim().x+M
  if blocksize+k<N;@inbounds cache[k+Int32(1)]=subject[blocksize+k+Int32(1)];end
  k+=blockDim().x
 end
 sync_threads()
 if idx<N-M+Int32(1)
  residues=0.f0;avg=@inbounds avgs[idx+Int32(1)];std=@inbounds stds[idx+Int32(1)]
  for i in Int32(0):M-Int32(1)
   value=((@inbounds cache[lid+i+Int32(1)])-avg)/std;low=value-(@inbounds lower[i+Int32(1)]);up=value-(@inbounds upper[i+Int32(1)])
   residues+=up*up*Float32(up>0.f0)+low*low*Float32(low<0.f0)
  end
  @inbounds out[idx+Int32(1)]=residues
 end
 return
end
function factory(subject,avgs,stds,out,lower,upper,M,N)
 return ()->(@cuda threads=256 blocks=cld(N-M+1,256) shmem=(M+256)*4 keogh_kernel!(subject,avgs,stds,out,lower,upper,Int32(M),Int32(N)))
end
if abspath(PROGRAM_FILE)==@__FILE__;keogh_study(factory);end
