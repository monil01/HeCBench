using CUDA,KernelAbstractions
import Atomix
const MODEL="ka"
@inline function atomic_add!(a,i,v);Atomix.@atomic :monotonic a[i]+=v;nothing;end
include("../study_device.jl")
@kernel function kernel_apply_step_function!(input,output,N)
 pos=Int32(@index(Global,Linear))-Int32(1)
 body_apply_step_function!(pos,Int32(4096),input,output,N)
end
function dispatch!(::Val{:apply_step_function},s,input,output,N)
 s.launches+=1
 kernel_apply_step_function!(CUDABackend(),64)(input,output,N;ndrange=4096)
 nothing
end
@kernel function kernel_makeError!(err,output,Y,N)
 pos=Int32(@index(Global,Linear))-Int32(1)
 body_makeError!(pos,Int32(10),err,output,Y,N)
end
function dispatch!(::Val{:makeError},s,err,output,Y,N)
 s.launches+=1
 kernel_makeError!(CUDABackend(),1)(err,output,Y,N;ndrange=10)
 nothing
end
@kernel function kernel_apply_grad!(output,grad,N)
 pos=Int32(@index(Global,Linear))-Int32(1)
 body_apply_grad!(pos,Int32(4096),output,grad,N)
end
function dispatch!(::Val{:apply_grad},s,output,grad,N)
 s.launches+=1
 kernel_apply_grad!(CUDABackend(),64)(output,grad,N;ndrange=4096)
 nothing
end
@kernel function kernel_fp_preact_c1!(input,preact,weight)
 pos=Int32(@index(Global,Linear))-Int32(1)
 body_fp_preact_c1!(pos,Int32(4096),input,preact,weight)
end
function dispatch!(::Val{:fp_preact_c1},s,input,preact,weight)
 s.launches+=1
 kernel_fp_preact_c1!(CUDABackend(),64)(input,preact,weight;ndrange=4096)
 nothing
end
@kernel function kernel_fp_bias_c1!(preact,bias)
 pos=Int32(@index(Global,Linear))-Int32(1)
 body_fp_bias_c1!(pos,Int32(4096),preact,bias)
end
function dispatch!(::Val{:fp_bias_c1},s,preact,bias)
 s.launches+=1
 kernel_fp_bias_c1!(CUDABackend(),64)(preact,bias;ndrange=4096)
 nothing
end
@kernel function kernel_fp_preact_s1!(input,preact,weight)
 pos=Int32(@index(Global,Linear))-Int32(1)
 body_fp_preact_s1!(pos,Int32(4096),input,preact,weight)
end
function dispatch!(::Val{:fp_preact_s1},s,input,preact,weight)
 s.launches+=1
 kernel_fp_preact_s1!(CUDABackend(),64)(input,preact,weight;ndrange=4096)
 nothing
end
@kernel function kernel_fp_bias_s1!(preact,bias)
 pos=Int32(@index(Global,Linear))-Int32(1)
 body_fp_bias_s1!(pos,Int32(4096),preact,bias)
end
function dispatch!(::Val{:fp_bias_s1},s,preact,bias)
 s.launches+=1
 kernel_fp_bias_s1!(CUDABackend(),64)(preact,bias;ndrange=4096)
 nothing
end
@kernel function kernel_fp_preact_f!(input,preact,weight)
 pos=Int32(@index(Global,Linear))-Int32(1)
 body_fp_preact_f!(pos,Int32(4096),input,preact,weight)
end
function dispatch!(::Val{:fp_preact_f},s,input,preact,weight)
 s.launches+=1
 kernel_fp_preact_f!(CUDABackend(),64)(input,preact,weight;ndrange=4096)
 nothing
end
@kernel function kernel_fp_bias_f!(preact,bias)
 pos=Int32(@index(Global,Linear))-Int32(1)
 body_fp_bias_f!(pos,Int32(4096),preact,bias)
end
function dispatch!(::Val{:fp_bias_f},s,preact,bias)
 s.launches+=1
 kernel_fp_bias_f!(CUDABackend(),64)(preact,bias;ndrange=4096)
 nothing
end
@kernel function kernel_bp_weight_f!(dweight,dpreact,poutput)
 pos=Int32(@index(Global,Linear))-Int32(1)
 body_bp_weight_f!(pos,Int32(4096),dweight,dpreact,poutput)
end
function dispatch!(::Val{:bp_weight_f},s,dweight,dpreact,poutput)
 s.launches+=1
 kernel_bp_weight_f!(CUDABackend(),64)(dweight,dpreact,poutput;ndrange=4096)
 nothing
end
@kernel function kernel_bp_bias_f!(bias,dpreact)
 pos=Int32(@index(Global,Linear))-Int32(1)
 body_bp_bias_f!(pos,Int32(4096),bias,dpreact)
end
function dispatch!(::Val{:bp_bias_f},s,bias,dpreact)
 s.launches+=1
 kernel_bp_bias_f!(CUDABackend(),64)(bias,dpreact;ndrange=4096)
 nothing
end
@kernel function kernel_bp_output_s1!(doutput,nweight,ndpreact)
 pos=Int32(@index(Global,Linear))-Int32(1)
 body_bp_output_s1!(pos,Int32(4096),doutput,nweight,ndpreact)
end
function dispatch!(::Val{:bp_output_s1},s,doutput,nweight,ndpreact)
 s.launches+=1
 kernel_bp_output_s1!(CUDABackend(),64)(doutput,nweight,ndpreact;ndrange=4096)
 nothing
end
@kernel function kernel_bp_preact_s1!(dpreact,doutput,preact)
 pos=Int32(@index(Global,Linear))-Int32(1)
 body_bp_preact_s1!(pos,Int32(4096),dpreact,doutput,preact)
end
function dispatch!(::Val{:bp_preact_s1},s,dpreact,doutput,preact)
 s.launches+=1
 kernel_bp_preact_s1!(CUDABackend(),64)(dpreact,doutput,preact;ndrange=4096)
 nothing
end
@kernel function kernel_bp_weight_s1!(dweight,dpreact,poutput)
 pos=Int32(@index(Global,Linear))-Int32(1)
 body_bp_weight_s1!(pos,Int32(4096),dweight,dpreact,poutput)
end
function dispatch!(::Val{:bp_weight_s1},s,dweight,dpreact,poutput)
 s.launches+=1
 kernel_bp_weight_s1!(CUDABackend(),64)(dweight,dpreact,poutput;ndrange=4096)
 nothing
end
@kernel function kernel_bp_bias_s1!(bias,dpreact)
 pos=Int32(@index(Global,Linear))-Int32(1)
 body_bp_bias_s1!(pos,Int32(4096),bias,dpreact)
end
function dispatch!(::Val{:bp_bias_s1},s,bias,dpreact)
 s.launches+=1
 kernel_bp_bias_s1!(CUDABackend(),64)(bias,dpreact;ndrange=4096)
 nothing
end
@kernel function kernel_bp_output_c1!(doutput,nweight,ndpreact)
 pos=Int32(@index(Global,Linear))-Int32(1)
 body_bp_output_c1!(pos,Int32(4096),doutput,nweight,ndpreact)
end
function dispatch!(::Val{:bp_output_c1},s,doutput,nweight,ndpreact)
 s.launches+=1
 kernel_bp_output_c1!(CUDABackend(),64)(doutput,nweight,ndpreact;ndrange=4096)
 nothing
end
@kernel function kernel_bp_preact_c1!(dpreact,doutput,preact)
 pos=Int32(@index(Global,Linear))-Int32(1)
 body_bp_preact_c1!(pos,Int32(4096),dpreact,doutput,preact)
end
function dispatch!(::Val{:bp_preact_c1},s,dpreact,doutput,preact)
 s.launches+=1
 kernel_bp_preact_c1!(CUDABackend(),64)(dpreact,doutput,preact;ndrange=4096)
 nothing
end
@kernel function kernel_bp_weight_c1!(dweight,dpreact,poutput)
 pos=Int32(@index(Global,Linear))-Int32(1)
 body_bp_weight_c1!(pos,Int32(4096),dweight,dpreact,poutput)
end
function dispatch!(::Val{:bp_weight_c1},s,dweight,dpreact,poutput)
 s.launches+=1
 kernel_bp_weight_c1!(CUDABackend(),64)(dweight,dpreact,poutput;ndrange=4096)
 nothing
end
@kernel function kernel_bp_bias_c1!(bias,dpreact)
 pos=Int32(@index(Global,Linear))-Int32(1)
 body_bp_bias_c1!(pos,Int32(4096),bias,dpreact)
end
function dispatch!(::Val{:bp_bias_c1},s,bias,dpreact)
 s.launches+=1
 kernel_bp_bias_c1!(CUDABackend(),64)(bias,dpreact;ndrange=4096)
 nothing
end
include("../study_common.jl")
main()
