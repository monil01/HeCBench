using CUDA
source=replace(read(joinpath(@__DIR__,"main.jl"),String),"exit(main(ARGS))"=>"")
include_string(Main,source,joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function prepare(din,dout,ddy,dx,batch,C,H,W,OH,OW)
    n=batch*C*H*W
    function launch!()
        @cuda threads=256 blocks=cld(n,256) pool2d_grad_kernel!(Int32(n),din,dout,ddy,Int32(C),Int32(H),Int32(W),Int32(OH),Int32(OW),dx)
        nothing
    end
    launch!
end
exit(study_main(prepare))
