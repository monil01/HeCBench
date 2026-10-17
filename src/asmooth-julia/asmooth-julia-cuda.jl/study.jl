include(joinpath(@__DIR__,"..","study_common.jl"))
s=read(joinpath(@__DIR__,"main.jl"),String);include_string(@__MODULE__,replace(s,r"exit\(main\(ARGS\)\)\s*$"=>""),joinpath(@__DIR__,"main.jl"))
function study_prepare(img,box,norm,out,w,t,r)
    function launch!()
        @cuda threads=(16,16) blocks=(8,8) smoothing_filter_kernel!(Int32(w),Int32(w),Int32(t),Int32(r),img,box,norm)
        @cuda threads=(16,16) blocks=(8,8) normalize_filter_kernel!(Int32(w),Int32(w),img,norm)
        @cuda threads=(16,16) blocks=(8,8) out_filter_kernel!(Int32(w),Int32(w),img,box,out)
    end
end
exit(study_main(study_prepare))
