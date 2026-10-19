using CUDA
include("../study_params.jl")
include("../device_common.jl")
allocate_model(x)=CuArray(x)
function gaussian!(data,out,rows,cols,coeff)
    lc=Int32(threadIdx().x);lr=Int32(threadIdx().y);gc=(Int32(blockIdx().x)-1)*16+lc;gr=(Int32(blockIdx().y)-1)*16+lr
    tile=CuDynamicSharedArray(Int32,324)
    load_tile!(tile,data,lr,lc,gr,gc,cols);sync_threads()
    gaussian_pixel!(tile,out,coeff,lr,lc,gr,gc,cols)
    return nothing
end
function sobel!(data,out,theta,rows,cols,sobx,soby)
    lc=Int32(threadIdx().x);lr=Int32(threadIdx().y);gc=(Int32(blockIdx().x)-1)*16+lc;gr=(Int32(blockIdx().y)-1)*16+lr
    tile=CuDynamicSharedArray(Int32,324)
    load_tile!(tile,data,lr,lc,gr,gc,cols);sync_threads()
    sobel_pixel!(tile,out,theta,sobx,soby,lr,lc,gr,gc,cols)
    return nothing
end
function nms!(data,out,theta,rows,cols)
    lc=Int32(threadIdx().x);lr=Int32(threadIdx().y);gc=(Int32(blockIdx().x)-1)*16+lc;gr=(Int32(blockIdx().y)-1)*16+lr
    tile=CuDynamicSharedArray(Int32,324)
    load_tile!(tile,data,lr,lc,gr,gc,cols);sync_threads()
    nms_pixel!(tile,out,theta,lr,lc,gr,gc,cols)
    return nothing
end
function threshold!(data,out,rows,cols)
    gc=(Int32(blockIdx().x)-1)*16+Int32(threadIdx().x);gr=(Int32(blockIdx().y)-1)*16+Int32(threadIdx().y)
    threshold_pixel!(data,out,gr,gc,cols)
    return nothing
end
function launch_stage!(k,w,coeff)
    a,b,t=w.device;s=w.s;rows=Int32(s.rows);cols=Int32(s.cols);blocks=((s.cols-2)÷16,(s.rows-2)÷16)
    if k==1
        @cuda threads=(16,16) blocks=blocks shmem=1296 gaussian!(a,b,rows,cols,coeff[1])
    elseif k==2
        @cuda threads=(16,16) blocks=blocks shmem=1296 sobel!(b,a,t,rows,cols,coeff[2],coeff[3])
    elseif k==3
        @cuda threads=(16,16) blocks=blocks shmem=1296 nms!(a,b,t,rows,cols)
    else
        @cuda threads=(16,16) blocks=blocks shmem=0 threshold!(b,a,rows,cols)
    end
end
include("../study_common.jl")
study_main("cuda_jl")
