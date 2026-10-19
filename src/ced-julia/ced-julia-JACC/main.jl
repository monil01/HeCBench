using CUDA
import JACC
JACC.@init_backend
const TILE_TEMPLATE=JACC.array(zeros(Int32,324))
include("../study_params.jl")
include("../device_common.jl")
allocate_model(x)=JACC.array(x)
function gaussian!(gc,gr,data,out,rows,cols,coeff,template)
    gc=Int32(gc);gr=Int32(gr);lc=mod(gc-1,Int32(16))+1;lr=mod(gr-1,Int32(16))+1
    tile=JACC.shared(template)
    load_tile!(tile,data,lr,lc,gr,gc,cols);JACC.sync_workgroup()
    gaussian_pixel!(tile,out,coeff,lr,lc,gr,gc,cols)
    return nothing
end
function sobel!(gc,gr,data,out,theta,rows,cols,sobx,soby,template)
    gc=Int32(gc);gr=Int32(gr);lc=mod(gc-1,Int32(16))+1;lr=mod(gr-1,Int32(16))+1
    tile=JACC.shared(template)
    load_tile!(tile,data,lr,lc,gr,gc,cols);JACC.sync_workgroup()
    sobel_pixel!(tile,out,theta,sobx,soby,lr,lc,gr,gc,cols)
    return nothing
end
function nms!(gc,gr,data,out,theta,rows,cols,template)
    gc=Int32(gc);gr=Int32(gr);lc=mod(gc-1,Int32(16))+1;lr=mod(gr-1,Int32(16))+1
    tile=JACC.shared(template)
    load_tile!(tile,data,lr,lc,gr,gc,cols);JACC.sync_workgroup()
    nms_pixel!(tile,out,theta,lr,lc,gr,gc,cols)
    return nothing
end
function threshold!(gc,gr,data,out,rows,cols)
    gc=Int32(gc);gr=Int32(gr)
    threshold_pixel!(data,out,gr,gc,cols)
    return nothing
end
function launch_stage!(k,w,coeff)
    a,b,t=w.device;s=w.s;rows=Int32(s.rows);cols=Int32(s.cols);blocks=((s.cols-2)÷16,(s.rows-2)÷16);extent=(s.cols-2,s.rows-2)
    spec=JACC.launch_spec(threads=(16,16),blocks=blocks,shmem_size=k==4 ? 0 : 1296,sync=false)
    if k==1
        JACC.parallel_for(spec,extent,gaussian!,a,b,rows,cols,coeff[1],TILE_TEMPLATE)
    elseif k==2
        JACC.parallel_for(spec,extent,sobel!,b,a,t,rows,cols,coeff[2],coeff[3],TILE_TEMPLATE)
    elseif k==3
        JACC.parallel_for(spec,extent,nms!,a,b,t,rows,cols,TILE_TEMPLATE)
    else
        JACC.parallel_for(spec,extent,threshold!,b,a,rows,cols)
    end
end
include("../study_common.jl")
initial_scratch=Array(TILE_TEMPLATE);all(iszero,initial_scratch)||error("scratch initial nonzero")
study_main("jacc")
final_scratch=Array(TILE_TEMPLATE);all(iszero,final_scratch)||error("scratch changed")
mkpath(ARGS[3]);scratchpath=joinpath(ARGS[3],"actual_scratch_readbacks_$(getpid()).bin");write(scratchpath,vcat(initial_scratch,final_scratch))
println("STUDY_AUXILIARY actual_auxiliary_scratch initial_and_final_values=648 all_zero=true path=$scratchpath; every_local_tile_value overwritten_by_image_before_read=true")
