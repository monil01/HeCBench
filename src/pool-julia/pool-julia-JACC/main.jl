using CUDA
using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function pool2d_grad_kernel!(g,nthreads::Int32, input_data, output_data, output_grad,
                             channels::Int32, input_h::Int32, input_w::Int32,
                             output_h::Int32, output_w::Int32, input_grad)
    index0 = g - 1
    if index0 < nthreads
        w_offset = index0 % input_w + ST_PAD_W
        tmp = index0 ÷ input_w
        h_offset = tmp % input_h + ST_PAD_H
        tmp ÷= input_h
        offset_c = tmp % channels
        batch_idx = tmp ÷ channels

        phstart = h_offset < ST_KSIZE_H ? Int32(0) : (h_offset - ST_KSIZE_H) ÷ ST_STRIDE_H + Int32(1)
        pwstart = w_offset < ST_KSIZE_W ? Int32(0) : (w_offset - ST_KSIZE_W) ÷ ST_STRIDE_W + Int32(1)
        phend = min(h_offset ÷ ST_STRIDE_H + Int32(1), output_h)
        pwend = min(w_offset ÷ ST_STRIDE_W + Int32(1), output_w)

        gradient = 0.0f0
        output_stride = batch_idx * output_h * output_w * channels +
                        offset_c * output_h * output_w
        for ph in phstart:phend-Int32(1)
            for pw in pwstart:pwend-Int32(1)
                hstart = ph * ST_STRIDE_H - ST_PAD_H
                wstart = pw * ST_STRIDE_W - ST_PAD_W
                hend = min(hstart + ST_KSIZE_H, input_h)
                wend = min(wstart + ST_KSIZE_W, input_w)
                hstart = max(hstart, Int32(0))
                wstart = max(wstart, Int32(0))
                pool_size = (hend - hstart) * (wend - wstart)
                output_sub_idx = ph * output_w + pw
                @inbounds gradient += (1.0f0 / Float32(pool_size)) *
                    output_grad[output_stride + output_sub_idx + Int32(1)]
            end
        end
        @inbounds input_grad[index0 + Int32(1)] = gradient
    end
end

function prepare(din,dout,ddy,dx,batch,C,H,W,OH,OW)
    n=batch*C*H*W;backend=JACC.get_backend(:cuda)
    spec=JACC.LaunchSpec{typeof(backend)}(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false)
    ()->JACC.parallel_for(pool2d_grad_kernel!,spec,n,Int32(n),din,dout,ddy,Int32(C),Int32(H),Int32(W),Int32(OH),Int32(OW),dx)
end
exit(study_main(prepare))
