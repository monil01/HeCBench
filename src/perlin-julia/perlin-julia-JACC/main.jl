using CUDA
import JACC
JACC.@init_backend
include("../device_math.jl")
include("../study_common.jl")
function perlin_jacc!(i,ystart,height,p,hash,gx,gy,out)
    tile=Int32((i-1)÷256);lane=Int32((i-1)%256)
    px=(tile%Int32(3840))*Int32(16)+lane%Int32(16)
    py=(tile÷Int32(3840))*Int32(16)+lane÷Int32(16)+ystart
    pixel_noise!(px,py,ystart,height,p,hash,gx,gy,out)
    nothing
end
function observer_jacc!(i,out,p,h,ns)
    if i==1;echo_noise!(out,p,h,ns);end
    nothing
end
function launch_jacc(stream,ystart,height,p,hash,gx,gy,out)
    blocks=3840*cld(Int(height),16)
    spec=JACC.launch_spec(stream=stream,threads=256,blocks=blocks,shmem_size=0,sync=false)
    JACC.parallel_for(spec,blocks*256,perlin_jacc!,ystart,height,p,hash,gx,gy,out)
    nothing
end
function observe_jacc(out,p,h,ns)
    spec=JACC.launch_spec(threads=32,blocks=1,shmem_size=0,sync=false)
    JACC.parallel_for(spec,32,observer_jacc!,out,p,h,ns)
    nothing
end
perlin_study(JACC.array,launch_jacc,observe_jacc)
