using CUDA,JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_common.jl"))
function keogh_jacc!(index,subject,avgs,stds,out,lower,upper,M::Int32,N::Int32,template)
 cache=JACC.shared(template);lid=(mod(Int32(index)-Int32(1),Int32(256))+Int32(1))-Int32(1);blocksize=Int32(256)*((div(Int32(index)-Int32(1),Int32(256))+Int32(1))-Int32(1));idx=blocksize+lid
 k=lid
 while k<Int32(256)+M
  if blocksize+k<N;@inbounds cache[k+Int32(1)]=subject[blocksize+k+Int32(1)];end
  k+=Int32(256)
 end
 JACC.sync_workgroup()
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
 blocks=cld(N-M+1,256);padded=blocks*256;template=CUDA.zeros(Float32,M+256)
 spec=JACC.launch_spec(;threads=256,blocks=blocks,shmem_size=(M+256)*4,sync=false)
 return ()->JACC.parallel_for(spec,padded,keogh_jacc!,subject,avgs,stds,out,lower,upper,Int32(M),Int32(N),template)
end
keogh_study(factory)
