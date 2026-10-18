using CUDA,KernelAbstractions
@inline fidx(i,j,b,s)=((i*41+j)*4+b)*2+s+2
@inline tidx(i,b,k,s)=((i*4+b)*2+k)*3+s+2
@inline lidx(i,j,b,s)=((i*2+j)*4+b)*2+s+2
@kernel function phmm_ka!(i,j,input,trans,emis,like,start,output)
 b=Int32(@index(Group,Linear))-Int32(1);state=Int32(@index(Local,Linear))-Int32(1)
 sh=@localmem Float64 (56,)
 @inbounds sh[b*2+state+1]=emis[fidx(i,j,b,state)]
 @synchronize
 # Native private duplicated transition/tensor materialization scalar-replaced.
 @synchronize
 s0=0.;s1=0.;s2=0.;s3=0.
 for k in Int32(0):Int32(1)
  @inbounds begin
   s0+=input[fidx(i-1,j-1,b,k)]*trans[tidx(i-1,b,k,state)]
   s1+=input[fidx(i-1,j,b,k)]*trans[tidx(i-1,b,k,state)]
   s2+=input[fidx(i,j-1,b,k)]*trans[tidx(i,b,k,state)]
   s3+=input[fidx(i,j,b,k)]*trans[tidx(i,b,k,state)]
  end
 end
 @inbounds begin
  sh[8+b*2+state+1]=s0*like[lidx(0,0,b,state)]
  sh[16+b*2+state+1]=s1*like[lidx(0,1,b,state)]
  sh[24+b*2+state+1]=s2*like[lidx(1,0,b,state)]
  sh[32+b*2+state+1]=s3*like[lidx(1,1,b,state)]
 end
 @synchronize
 # Preserve source duplicate identical writes of both outputstates by boththreads.
 for s in Int32(0):Int32(1)
  @inbounds output[fidx(i,j,b,s)]=(((sh[8+b*2+s+1]+sh[16+b*2+s+1])+sh[24+b*2+s+1])+sh[32+b*2+s+1])*sh[b*2+s+1]
 end
 nothing
end
allocate(v)=begin
 a=KernelAbstractions.allocate(CUDA.CUDABackend(),Float64,size(v));copyto!(a,v);a
end
function launch(i,j,c,src,dst)
 phmm_ka!(KernelAbstractions.get_backend(dst),2)(Int32(i),Int32(j),src,c.arrays[4],c.arrays[3],c.arrays[5],c.arrays[6],dst;ndrange=8)
 nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(allocate,launch)
