CUDA.device!(0);CUDA.context();CUDA.stream!(CUDA.legacy_stream())
const MCMD_DISPATCHES=Ref(0)
function force_callback(ptr::Ptr{Cdouble},n::Cint,grid::Cint,block::Cint,pf::Cint,cut::Cdouble,alpha::Cdouble,km::Cint,ks::Cint,damp::Cdouble,bp::Ptr{Cdouble},rp::Ptr{Cdouble})::Cint
    try
        pf==1||error("Uncertifiedpotentialmode: namedpipelineispform1LJ+Ewald")
        a=unsafe_wrap(CuArray,CuPtr{Float64}(UInt(ptr)),(14*Int(n),);own=false)
        b=unsafe_wrap(CuArray,CuPtr{Float64}(UInt(bp)),(9,);own=false)
        rb=unsafe_wrap(CuArray,CuPtr{Float64}(UInt(rp)),(9,);own=false)
        mcmd_launch(a,Int32(n),Int(grid),Int(block),Float64(cut),b,rb,Float64(alpha),Int32(km),Int32(ks))
        MCMD_DISPATCHES[]+=1
        return Cint(0)
    catch err
        showerror(stderr,err,catch_backtrace());println(stderr);flush(stderr)
        return Cint(1)
    end
end
const MCMD_CALLBACK=@cfunction(force_callback,Cint,(Ptr{Cdouble},Cint,Cint,Cint,Cint,Cdouble,Cdouble,Cint,Cint,Cdouble,Ptr{Cdouble},Ptr{Cdouble}))
function mcmd_study()
    path=abspath(ARGS[1]);lib=Libdl.dlopen(path);run=Libdl.dlsym(lib,:mcmd_run)
    println("STUDY_RUNTIME_BINDING wrapper_sha256=",bytes2hex(sha256(read(path)))," original_CPU_MDcontroller=true explicit_native_kernel_fallback=false")
    status=ccall(run,Cint,(Cstring,Ptr{Cvoid}),abspath(ARGS[2]),MCMD_CALLBACK)
    CUDA.device_synchronize();println("STUDY_ACTUAL_AUTHORED_DISPATCHES ",MCMD_DISPATCHES[])
    status==0||error("FulloriginalMDstudyfaileddeclaredchecks/status ",status)
    MCMD_DISPATCHES[]==66||error("Expected11launchesperpipeline timeswarmup+fivecheckedtrials")
end
