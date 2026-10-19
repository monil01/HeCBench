using CUDA, KernelAbstractions
include("../device_math.jl")
include("../study_common.jl")
@kernel function perlin_ka!(ystart,height,p,hash,gx,gy,out)
    group=@index(Group,Linear)
    local_id=@index(Local,Linear)
    tile=Int32(group-1);lane=Int32(local_id-1)
    px=(tile%Int32(3840))*Int32(16)+lane%Int32(16)
    py=(tile÷Int32(3840))*Int32(16)+lane÷Int32(16)+ystart
    pixel_noise!(px,py,ystart,height,p,hash,gx,gy,out)
end
@kernel function observer_ka!(out,p,h,ns)
    i=@index(Global,Linear)
    if i==1;echo_noise!(out,p,h,ns);end
end
const perlin_backend=CUDA.CUDABackend()
function launch_ka(stream,ystart,height,p,hash,gx,gy,out)
    previous=CUDA.stream()
    CUDA.stream!(stream)
    try
        perlin_ka!(perlin_backend,256)(ystart,height,p,hash,gx,gy,out;ndrange=3840*cld(Int(height),16)*256)
    finally
        CUDA.stream!(previous)
    end
    nothing
end
observe_ka(out,p,h,ns)=(observer_ka!(perlin_backend,32)(out,p,h,ns;ndrange=32);nothing)
perlin_study(a->KernelAbstractions.allocate(perlin_backend,eltype(a),size(a)) |> d->(copyto!(d,a);d),launch_ka,observe_ka)
