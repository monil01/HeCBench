using CUDA
import JACC
JACC.@init_backend
const MODEL="jacc"
@inline function atomic_add!(a,i,v);JACC.@atomic a[i]+=v;nothing;end
include("../study_device.jl")
function kernel_apply_step_function!(index,input,output,N)
 pos=Int32(index)-Int32(1)
 body_apply_step_function!(pos,Int32(4096),input,output,N)
 nothing
end
function dispatch!(::Val{:apply_step_function},s,input,output,N)
 s.launches+=1
 spec=JACC.launch_spec(threads=64,blocks=64,shmem_size=0,sync=false)
 JACC.parallel_for(spec,4096,kernel_apply_step_function!,input,output,N)
 nothing
end
function kernel_makeError!(index,err,output,Y,N)
 pos=Int32(index)-Int32(1)
 body_makeError!(pos,Int32(10),err,output,Y,N)
 nothing
end
function dispatch!(::Val{:makeError},s,err,output,Y,N)
 s.launches+=1
 spec=JACC.launch_spec(threads=1,blocks=10,shmem_size=0,sync=false)
 JACC.parallel_for(spec,10,kernel_makeError!,err,output,Y,N)
 nothing
end
function kernel_apply_grad!(index,output,grad,N)
 pos=Int32(index)-Int32(1)
 body_apply_grad!(pos,Int32(4096),output,grad,N)
 nothing
end
function dispatch!(::Val{:apply_grad},s,output,grad,N)
 s.launches+=1
 spec=JACC.launch_spec(threads=64,blocks=64,shmem_size=0,sync=false)
 JACC.parallel_for(spec,4096,kernel_apply_grad!,output,grad,N)
 nothing
end
function kernel_fp_preact_c1!(index,input,preact,weight)
 pos=Int32(index)-Int32(1)
 body_fp_preact_c1!(pos,Int32(4096),input,preact,weight)
 nothing
end
function dispatch!(::Val{:fp_preact_c1},s,input,preact,weight)
 s.launches+=1
 spec=JACC.launch_spec(threads=64,blocks=64,shmem_size=0,sync=false)
 JACC.parallel_for(spec,4096,kernel_fp_preact_c1!,input,preact,weight)
 nothing
end
function kernel_fp_bias_c1!(index,preact,bias)
 pos=Int32(index)-Int32(1)
 body_fp_bias_c1!(pos,Int32(4096),preact,bias)
 nothing
end
function dispatch!(::Val{:fp_bias_c1},s,preact,bias)
 s.launches+=1
 spec=JACC.launch_spec(threads=64,blocks=64,shmem_size=0,sync=false)
 JACC.parallel_for(spec,4096,kernel_fp_bias_c1!,preact,bias)
 nothing
end
function kernel_fp_preact_s1!(index,input,preact,weight)
 pos=Int32(index)-Int32(1)
 body_fp_preact_s1!(pos,Int32(4096),input,preact,weight)
 nothing
end
function dispatch!(::Val{:fp_preact_s1},s,input,preact,weight)
 s.launches+=1
 spec=JACC.launch_spec(threads=64,blocks=64,shmem_size=0,sync=false)
 JACC.parallel_for(spec,4096,kernel_fp_preact_s1!,input,preact,weight)
 nothing
end
function kernel_fp_bias_s1!(index,preact,bias)
 pos=Int32(index)-Int32(1)
 body_fp_bias_s1!(pos,Int32(4096),preact,bias)
 nothing
end
function dispatch!(::Val{:fp_bias_s1},s,preact,bias)
 s.launches+=1
 spec=JACC.launch_spec(threads=64,blocks=64,shmem_size=0,sync=false)
 JACC.parallel_for(spec,4096,kernel_fp_bias_s1!,preact,bias)
 nothing
end
function kernel_fp_preact_f!(index,input,preact,weight)
 pos=Int32(index)-Int32(1)
 body_fp_preact_f!(pos,Int32(4096),input,preact,weight)
 nothing
end
function dispatch!(::Val{:fp_preact_f},s,input,preact,weight)
 s.launches+=1
 spec=JACC.launch_spec(threads=64,blocks=64,shmem_size=0,sync=false)
 JACC.parallel_for(spec,4096,kernel_fp_preact_f!,input,preact,weight)
 nothing
end
function kernel_fp_bias_f!(index,preact,bias)
 pos=Int32(index)-Int32(1)
 body_fp_bias_f!(pos,Int32(4096),preact,bias)
 nothing
end
function dispatch!(::Val{:fp_bias_f},s,preact,bias)
 s.launches+=1
 spec=JACC.launch_spec(threads=64,blocks=64,shmem_size=0,sync=false)
 JACC.parallel_for(spec,4096,kernel_fp_bias_f!,preact,bias)
 nothing
end
function kernel_bp_weight_f!(index,dweight,dpreact,poutput)
 pos=Int32(index)-Int32(1)
 body_bp_weight_f!(pos,Int32(4096),dweight,dpreact,poutput)
 nothing
end
function dispatch!(::Val{:bp_weight_f},s,dweight,dpreact,poutput)
 s.launches+=1
 spec=JACC.launch_spec(threads=64,blocks=64,shmem_size=0,sync=false)
 JACC.parallel_for(spec,4096,kernel_bp_weight_f!,dweight,dpreact,poutput)
 nothing
end
function kernel_bp_bias_f!(index,bias,dpreact)
 pos=Int32(index)-Int32(1)
 body_bp_bias_f!(pos,Int32(4096),bias,dpreact)
 nothing
end
function dispatch!(::Val{:bp_bias_f},s,bias,dpreact)
 s.launches+=1
 spec=JACC.launch_spec(threads=64,blocks=64,shmem_size=0,sync=false)
 JACC.parallel_for(spec,4096,kernel_bp_bias_f!,bias,dpreact)
 nothing
end
function kernel_bp_output_s1!(index,doutput,nweight,ndpreact)
 pos=Int32(index)-Int32(1)
 body_bp_output_s1!(pos,Int32(4096),doutput,nweight,ndpreact)
 nothing
end
function dispatch!(::Val{:bp_output_s1},s,doutput,nweight,ndpreact)
 s.launches+=1
 spec=JACC.launch_spec(threads=64,blocks=64,shmem_size=0,sync=false)
 JACC.parallel_for(spec,4096,kernel_bp_output_s1!,doutput,nweight,ndpreact)
 nothing
end
function kernel_bp_preact_s1!(index,dpreact,doutput,preact)
 pos=Int32(index)-Int32(1)
 body_bp_preact_s1!(pos,Int32(4096),dpreact,doutput,preact)
 nothing
end
function dispatch!(::Val{:bp_preact_s1},s,dpreact,doutput,preact)
 s.launches+=1
 spec=JACC.launch_spec(threads=64,blocks=64,shmem_size=0,sync=false)
 JACC.parallel_for(spec,4096,kernel_bp_preact_s1!,dpreact,doutput,preact)
 nothing
end
function kernel_bp_weight_s1!(index,dweight,dpreact,poutput)
 pos=Int32(index)-Int32(1)
 body_bp_weight_s1!(pos,Int32(4096),dweight,dpreact,poutput)
 nothing
end
function dispatch!(::Val{:bp_weight_s1},s,dweight,dpreact,poutput)
 s.launches+=1
 spec=JACC.launch_spec(threads=64,blocks=64,shmem_size=0,sync=false)
 JACC.parallel_for(spec,4096,kernel_bp_weight_s1!,dweight,dpreact,poutput)
 nothing
end
function kernel_bp_bias_s1!(index,bias,dpreact)
 pos=Int32(index)-Int32(1)
 body_bp_bias_s1!(pos,Int32(4096),bias,dpreact)
 nothing
end
function dispatch!(::Val{:bp_bias_s1},s,bias,dpreact)
 s.launches+=1
 spec=JACC.launch_spec(threads=64,blocks=64,shmem_size=0,sync=false)
 JACC.parallel_for(spec,4096,kernel_bp_bias_s1!,bias,dpreact)
 nothing
end
function kernel_bp_output_c1!(index,doutput,nweight,ndpreact)
 pos=Int32(index)-Int32(1)
 body_bp_output_c1!(pos,Int32(4096),doutput,nweight,ndpreact)
 nothing
end
function dispatch!(::Val{:bp_output_c1},s,doutput,nweight,ndpreact)
 s.launches+=1
 spec=JACC.launch_spec(threads=64,blocks=64,shmem_size=0,sync=false)
 JACC.parallel_for(spec,4096,kernel_bp_output_c1!,doutput,nweight,ndpreact)
 nothing
end
function kernel_bp_preact_c1!(index,dpreact,doutput,preact)
 pos=Int32(index)-Int32(1)
 body_bp_preact_c1!(pos,Int32(4096),dpreact,doutput,preact)
 nothing
end
function dispatch!(::Val{:bp_preact_c1},s,dpreact,doutput,preact)
 s.launches+=1
 spec=JACC.launch_spec(threads=64,blocks=64,shmem_size=0,sync=false)
 JACC.parallel_for(spec,4096,kernel_bp_preact_c1!,dpreact,doutput,preact)
 nothing
end
function kernel_bp_weight_c1!(index,dweight,dpreact,poutput)
 pos=Int32(index)-Int32(1)
 body_bp_weight_c1!(pos,Int32(4096),dweight,dpreact,poutput)
 nothing
end
function dispatch!(::Val{:bp_weight_c1},s,dweight,dpreact,poutput)
 s.launches+=1
 spec=JACC.launch_spec(threads=64,blocks=64,shmem_size=0,sync=false)
 JACC.parallel_for(spec,4096,kernel_bp_weight_c1!,dweight,dpreact,poutput)
 nothing
end
function kernel_bp_bias_c1!(index,bias,dpreact)
 pos=Int32(index)-Int32(1)
 body_bp_bias_c1!(pos,Int32(4096),bias,dpreact)
 nothing
end
function dispatch!(::Val{:bp_bias_c1},s,bias,dpreact)
 s.launches+=1
 spec=JACC.launch_spec(threads=64,blocks=64,shmem_size=0,sync=false)
 JACC.parallel_for(spec,4096,kernel_bp_bias_c1!,bias,dpreact)
 nothing
end
include("../study_common.jl")
main()
