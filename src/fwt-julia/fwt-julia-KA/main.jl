using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function fwt_stage_ka!(data,stride::Int32,pairs::Int32)
 p=Int32(@index(Global,Linear))-Int32(1)
 if p<pairs
  base=div(p,stride)*(stride<<Int32(1));j=p-div(p,stride)*stride
  i0=base+j+Int32(1);i1=i0+stride
  a=@inbounds data[i0];b=@inbounds data[i1]
  @inbounds data[i0]=a+b;data[i1]=a-b
 end
end
@kernel function modulate_ka!(a,b,n::Int32,rcpn::Float32)
 i=Int32(@index(Global,Linear))
 if i<=n;@inbounds a[i]=a[i]*b[i]*rcpn;end
end
function factory(data,kernel,n)
 stage=fwt_stage_ka!(get_backend(data),256);modulate=modulate_ka!(get_backend(data),256);pairs=Int32(div(n,2))
 function transform!(x)
  stride=Int32(div(n,2))
  while stride>=1
   stage(x,stride,pairs;ndrange=Int(pairs));stride=div(stride,Int32(2))
  end
 end
 return ()->begin
  transform!(data);transform!(kernel)
  modulate(data,kernel,Int32(n),Float32(1/n);ndrange=n)
  transform!(data)
 end
end
fwt_study(factory)
