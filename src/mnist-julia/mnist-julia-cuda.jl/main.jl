using CUDA
const MODEL="cuda_jl"
@inline function atomic_add!(a,i,v);CUDA.@atomic a[i]+=v;nothing;end
include("../study_device.jl")
function kernel_apply_step_function!(input,output,N)
 pos=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+Int32(threadIdx().x)-Int32(1)
 size=Int32(blockDim().x)*Int32(gridDim().x)
 body_apply_step_function!(pos,size,input,output,N)
 nothing
end
function dispatch!(::Val{:apply_step_function},s,input,output,N)
 s.launches+=1
 CUDA.@cuda threads=64 blocks=64 kernel_apply_step_function!(input,output,N)
 nothing
end
function kernel_makeError!(err,output,Y,N)
 pos=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+Int32(threadIdx().x)-Int32(1)
 size=Int32(blockDim().x)*Int32(gridDim().x)
 body_makeError!(pos,size,err,output,Y,N)
 nothing
end
function dispatch!(::Val{:makeError},s,err,output,Y,N)
 s.launches+=1
 CUDA.@cuda threads=1 blocks=10 kernel_makeError!(err,output,Y,N)
 nothing
end
function kernel_apply_grad!(output,grad,N)
 pos=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+Int32(threadIdx().x)-Int32(1)
 size=Int32(blockDim().x)*Int32(gridDim().x)
 body_apply_grad!(pos,size,output,grad,N)
 nothing
end
function dispatch!(::Val{:apply_grad},s,output,grad,N)
 s.launches+=1
 CUDA.@cuda threads=64 blocks=64 kernel_apply_grad!(output,grad,N)
 nothing
end
function kernel_fp_preact_c1!(input,preact,weight)
 pos=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+Int32(threadIdx().x)-Int32(1)
 size=Int32(blockDim().x)*Int32(gridDim().x)
 body_fp_preact_c1!(pos,size,input,preact,weight)
 nothing
end
function dispatch!(::Val{:fp_preact_c1},s,input,preact,weight)
 s.launches+=1
 CUDA.@cuda threads=64 blocks=64 kernel_fp_preact_c1!(input,preact,weight)
 nothing
end
function kernel_fp_bias_c1!(preact,bias)
 pos=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+Int32(threadIdx().x)-Int32(1)
 size=Int32(blockDim().x)*Int32(gridDim().x)
 body_fp_bias_c1!(pos,size,preact,bias)
 nothing
end
function dispatch!(::Val{:fp_bias_c1},s,preact,bias)
 s.launches+=1
 CUDA.@cuda threads=64 blocks=64 kernel_fp_bias_c1!(preact,bias)
 nothing
end
function kernel_fp_preact_s1!(input,preact,weight)
 pos=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+Int32(threadIdx().x)-Int32(1)
 size=Int32(blockDim().x)*Int32(gridDim().x)
 body_fp_preact_s1!(pos,size,input,preact,weight)
 nothing
end
function dispatch!(::Val{:fp_preact_s1},s,input,preact,weight)
 s.launches+=1
 CUDA.@cuda threads=64 blocks=64 kernel_fp_preact_s1!(input,preact,weight)
 nothing
end
function kernel_fp_bias_s1!(preact,bias)
 pos=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+Int32(threadIdx().x)-Int32(1)
 size=Int32(blockDim().x)*Int32(gridDim().x)
 body_fp_bias_s1!(pos,size,preact,bias)
 nothing
end
function dispatch!(::Val{:fp_bias_s1},s,preact,bias)
 s.launches+=1
 CUDA.@cuda threads=64 blocks=64 kernel_fp_bias_s1!(preact,bias)
 nothing
end
function kernel_fp_preact_f!(input,preact,weight)
 pos=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+Int32(threadIdx().x)-Int32(1)
 size=Int32(blockDim().x)*Int32(gridDim().x)
 body_fp_preact_f!(pos,size,input,preact,weight)
 nothing
end
function dispatch!(::Val{:fp_preact_f},s,input,preact,weight)
 s.launches+=1
 CUDA.@cuda threads=64 blocks=64 kernel_fp_preact_f!(input,preact,weight)
 nothing
end
function kernel_fp_bias_f!(preact,bias)
 pos=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+Int32(threadIdx().x)-Int32(1)
 size=Int32(blockDim().x)*Int32(gridDim().x)
 body_fp_bias_f!(pos,size,preact,bias)
 nothing
end
function dispatch!(::Val{:fp_bias_f},s,preact,bias)
 s.launches+=1
 CUDA.@cuda threads=64 blocks=64 kernel_fp_bias_f!(preact,bias)
 nothing
end
function kernel_bp_weight_f!(dweight,dpreact,poutput)
 pos=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+Int32(threadIdx().x)-Int32(1)
 size=Int32(blockDim().x)*Int32(gridDim().x)
 body_bp_weight_f!(pos,size,dweight,dpreact,poutput)
 nothing
end
function dispatch!(::Val{:bp_weight_f},s,dweight,dpreact,poutput)
 s.launches+=1
 CUDA.@cuda threads=64 blocks=64 kernel_bp_weight_f!(dweight,dpreact,poutput)
 nothing
end
function kernel_bp_bias_f!(bias,dpreact)
 pos=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+Int32(threadIdx().x)-Int32(1)
 size=Int32(blockDim().x)*Int32(gridDim().x)
 body_bp_bias_f!(pos,size,bias,dpreact)
 nothing
end
function dispatch!(::Val{:bp_bias_f},s,bias,dpreact)
 s.launches+=1
 CUDA.@cuda threads=64 blocks=64 kernel_bp_bias_f!(bias,dpreact)
 nothing
end
function kernel_bp_output_s1!(doutput,nweight,ndpreact)
 pos=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+Int32(threadIdx().x)-Int32(1)
 size=Int32(blockDim().x)*Int32(gridDim().x)
 body_bp_output_s1!(pos,size,doutput,nweight,ndpreact)
 nothing
end
function dispatch!(::Val{:bp_output_s1},s,doutput,nweight,ndpreact)
 s.launches+=1
 CUDA.@cuda threads=64 blocks=64 kernel_bp_output_s1!(doutput,nweight,ndpreact)
 nothing
end
function kernel_bp_preact_s1!(dpreact,doutput,preact)
 pos=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+Int32(threadIdx().x)-Int32(1)
 size=Int32(blockDim().x)*Int32(gridDim().x)
 body_bp_preact_s1!(pos,size,dpreact,doutput,preact)
 nothing
end
function dispatch!(::Val{:bp_preact_s1},s,dpreact,doutput,preact)
 s.launches+=1
 CUDA.@cuda threads=64 blocks=64 kernel_bp_preact_s1!(dpreact,doutput,preact)
 nothing
end
function kernel_bp_weight_s1!(dweight,dpreact,poutput)
 pos=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+Int32(threadIdx().x)-Int32(1)
 size=Int32(blockDim().x)*Int32(gridDim().x)
 body_bp_weight_s1!(pos,size,dweight,dpreact,poutput)
 nothing
end
function dispatch!(::Val{:bp_weight_s1},s,dweight,dpreact,poutput)
 s.launches+=1
 CUDA.@cuda threads=64 blocks=64 kernel_bp_weight_s1!(dweight,dpreact,poutput)
 nothing
end
function kernel_bp_bias_s1!(bias,dpreact)
 pos=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+Int32(threadIdx().x)-Int32(1)
 size=Int32(blockDim().x)*Int32(gridDim().x)
 body_bp_bias_s1!(pos,size,bias,dpreact)
 nothing
end
function dispatch!(::Val{:bp_bias_s1},s,bias,dpreact)
 s.launches+=1
 CUDA.@cuda threads=64 blocks=64 kernel_bp_bias_s1!(bias,dpreact)
 nothing
end
function kernel_bp_output_c1!(doutput,nweight,ndpreact)
 pos=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+Int32(threadIdx().x)-Int32(1)
 size=Int32(blockDim().x)*Int32(gridDim().x)
 body_bp_output_c1!(pos,size,doutput,nweight,ndpreact)
 nothing
end
function dispatch!(::Val{:bp_output_c1},s,doutput,nweight,ndpreact)
 s.launches+=1
 CUDA.@cuda threads=64 blocks=64 kernel_bp_output_c1!(doutput,nweight,ndpreact)
 nothing
end
function kernel_bp_preact_c1!(dpreact,doutput,preact)
 pos=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+Int32(threadIdx().x)-Int32(1)
 size=Int32(blockDim().x)*Int32(gridDim().x)
 body_bp_preact_c1!(pos,size,dpreact,doutput,preact)
 nothing
end
function dispatch!(::Val{:bp_preact_c1},s,dpreact,doutput,preact)
 s.launches+=1
 CUDA.@cuda threads=64 blocks=64 kernel_bp_preact_c1!(dpreact,doutput,preact)
 nothing
end
function kernel_bp_weight_c1!(dweight,dpreact,poutput)
 pos=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+Int32(threadIdx().x)-Int32(1)
 size=Int32(blockDim().x)*Int32(gridDim().x)
 body_bp_weight_c1!(pos,size,dweight,dpreact,poutput)
 nothing
end
function dispatch!(::Val{:bp_weight_c1},s,dweight,dpreact,poutput)
 s.launches+=1
 CUDA.@cuda threads=64 blocks=64 kernel_bp_weight_c1!(dweight,dpreact,poutput)
 nothing
end
function kernel_bp_bias_c1!(bias,dpreact)
 pos=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+Int32(threadIdx().x)-Int32(1)
 size=Int32(blockDim().x)*Int32(gridDim().x)
 body_bp_bias_c1!(pos,size,bias,dpreact)
 nothing
end
function dispatch!(::Val{:bp_bias_c1},s,bias,dpreact)
 s.launches+=1
 CUDA.@cuda threads=64 blocks=64 kernel_bp_bias_c1!(bias,dpreact)
 nothing
end
include("../study_common.jl")
main()
