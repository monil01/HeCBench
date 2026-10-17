using CUDA
function gabor_kernel!(out,height,width,center_y,center_x,ctheta,stheta,scale,sx2,sy2,fx)
    x=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)-Int32(1);y=Int32((blockIdx().y-1)*blockDim().y+threadIdx().y)-Int32(1)
    if x<width&&y<height
        centered_x=Float64(x)-center_x;centered_y=Float64(y)-center_y
        u=ctheta*centered_x-stheta*centered_y;v=ctheta*centered_y+stheta*centered_x
        @inbounds out[y*width+x+Int32(1)]=scale*exp(-0.5*(u*u/sx2+v*v/sy2))*cos(2.0*pi*fx*u)
    end
    nothing
end
function study_prepare(out,h,w,cy,cx,ct,st,scale,sx2,sy2,fx)
    ()->(@cuda threads=(16,16) blocks=(cld(w,16),cld(h,16)) gabor_kernel!(out,Int32(h),Int32(w),cy,cx,ct,st,scale,sx2,sy2,fx))
end
if abspath(PROGRAM_FILE)==@__FILE__
    include(joinpath(@__DIR__,"..","study_common.jl"));exit(study_main(study_prepare))
end
