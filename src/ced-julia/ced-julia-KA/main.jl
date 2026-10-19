using CUDA, KernelAbstractions
include("../study_params.jl")
include("../device_common.jl")
allocate_model(x)=begin
    a=KernelAbstractions.allocate(CUDABackend(),eltype(x),size(x)...);copyto!(a,x);a
end
@kernel function gaussian!(data,out,rows,cols,coeff)
    lc=Int32(@index(Local,NTuple)[1]);lr=Int32(@index(Local,NTuple)[2]);gc=(Int32(@index(Group,NTuple)[1])-1)*16+lc;gr=(Int32(@index(Group,NTuple)[2])-1)*16+lr
    tile=@localmem Int32 (324,)
    load_tile!(tile,data,lr,lc,gr,gc,cols);@synchronize()
    gaussian_pixel!(tile,out,coeff,lr,lc,gr,gc,cols)
end
@kernel function sobel!(data,out,theta,rows,cols,sobx,soby)
    lc=Int32(@index(Local,NTuple)[1]);lr=Int32(@index(Local,NTuple)[2]);gc=(Int32(@index(Group,NTuple)[1])-1)*16+lc;gr=(Int32(@index(Group,NTuple)[2])-1)*16+lr
    tile=@localmem Int32 (324,)
    load_tile!(tile,data,lr,lc,gr,gc,cols);@synchronize()
    sobel_pixel!(tile,out,theta,sobx,soby,lr,lc,gr,gc,cols)
end
@kernel function nms!(data,out,theta,rows,cols)
    lc=Int32(@index(Local,NTuple)[1]);lr=Int32(@index(Local,NTuple)[2]);gc=(Int32(@index(Group,NTuple)[1])-1)*16+lc;gr=(Int32(@index(Group,NTuple)[2])-1)*16+lr
    tile=@localmem Int32 (324,)
    load_tile!(tile,data,lr,lc,gr,gc,cols);@synchronize()
    nms_pixel!(tile,out,theta,lr,lc,gr,gc,cols)
end
@kernel function threshold!(data,out,rows,cols)
    gc=Int32(@index(Global,NTuple)[1]);gr=Int32(@index(Global,NTuple)[2])
    threshold_pixel!(data,out,gr,gc,cols)
end
function launch_stage!(k,w,coeff)
    a,b,t=w.device;s=w.s;rows=Int32(s.rows);cols=Int32(s.cols);extent=(s.cols-2,s.rows-2);backend=CUDABackend()
    if k==1
        gaussian!(backend,(16,16))(a,b,rows,cols,coeff[1];ndrange=extent)
    elseif k==2
        sobel!(backend,(16,16))(b,a,t,rows,cols,coeff[2],coeff[3];ndrange=extent)
    elseif k==3
        nms!(backend,(16,16))(a,b,t,rows,cols;ndrange=extent)
    else
        threshold!(backend,(16,16))(b,a,rows,cols;ndrange=extent)
    end
end
include("../study_common.jl")
study_main("ka")
