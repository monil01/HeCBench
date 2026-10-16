include(joinpath(@__DIR__,"..","study_common.jl"))
s=read(joinpath(@__DIR__,"main.jl"),String);include_string(@__MODULE__,replace(s,r"exit\(main\(ARGS\)\)\s*$"=>""),joinpath(@__DIR__,"main.jl"))
function study_prepare(u,v,un,vn,w,h,nu,dt,dx,dy)
    function launch!()
        @cuda threads=(16,16) blocks=(cld(w-2,16),cld(h-2,16)) core_kernel!(un,vn,u,v,Int32(w),Int32(h),nu,dt,dx,dy)
        @cuda threads=256 blocks=cld(w,256) bound_h_kernel!(un,vn,Int32(w),Int32(h))
        @cuda threads=256 blocks=cld(h,256) bound_v_kernel!(un,vn,Int32(w),Int32(h))
        @cuda threads=256 blocks=cld(w*h,256) update_kernel!(u,v,un,vn,Int32(w*h))
    end
end
exit(study_main(study_prepare))
