using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function keogh_ka!(subject,avgs,stds,out,lower,upper,M::Int32,N::Int32,::Val{Q}) where {Q}
 cache=@localmem Float32 (Q+256,);lid=Int32(@index(Local,Linear))-Int32(1);blocksize=Int32(256)*(Int32(@index(Group,Linear))-Int32(1));idx=blocksize+lid
 k=lid
 while k<Int32(256)+M
  if blocksize+k<N;@inbounds cache[k+Int32(1)]=subject[blocksize+k+Int32(1)];end
  k+=Int32(256)
 end
 @synchronize
 if idx<N-M+Int32(1)
  residues=0.f0;avg=@inbounds avgs[idx+Int32(1)];std=@inbounds stds[idx+Int32(1)]
  for i in Int32(0):M-Int32(1)
   value=((@inbounds cache[lid+i+Int32(1)])-avg)/std;low=value-(@inbounds lower[i+Int32(1)]);up=value-(@inbounds upper[i+Int32(1)])
   residues+=up*up*Float32(up>0.f0)+low*low*Float32(low<0.f0)
  end
  @inbounds out[idx+Int32(1)]=residues
 end
end
function factory(subject,avgs,stds,out,lower,upper,M,N)
 launch=keogh_ka!(get_backend(subject),256);tag=Val(M);padded=cld(N-M+1,256)*256
 return ()->launch(subject,avgs,stds,out,lower,upper,Int32(M),Int32(N),tag;ndrange=padded)
end
keogh_study(factory)
