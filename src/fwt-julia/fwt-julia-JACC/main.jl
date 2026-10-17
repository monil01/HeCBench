using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function fwt_stage_jacc!(index,data,stride::Int32,pairs::Int32)
 p=Int32(index)-Int32(1)
 if p<pairs
  base=div(p,stride)*(stride<<Int32(1));j=p-div(p,stride)*stride
  i0=base+j+Int32(1);i1=i0+stride
  a=@inbounds data[i0];b=@inbounds data[i1]
  @inbounds data[i0]=a+b;data[i1]=a-b
 end
end
function modulate_jacc!(index,a,b,n::Int32,rcpn::Float32)
 i=Int32(index)
 if i<=n;@inbounds a[i]=a[i]*b[i]*rcpn;end
end
function factory(data,kernel,n)
 stage_spec=JACC.launch_spec(;threads=256,blocks=div(n,512),shmem_size=0,sync=false);mod_spec=JACC.launch_spec(;threads=256,blocks=div(n,256),shmem_size=0,sync=false);pairs=Int32(div(n,2))
 function transform!(x)
  stride=Int32(div(n,2))
  while stride>=1
   JACC.parallel_for(stage_spec,Int(pairs),fwt_stage_jacc!,x,stride,pairs);stride=div(stride,Int32(2))
  end
 end
 return ()->begin
  transform!(data);transform!(kernel)
  JACC.parallel_for(mod_spec,n,modulate_jacc!,data,kernel,Int32(n),Float32(1/n))
  transform!(data)
 end
end
fwt_study(factory)
