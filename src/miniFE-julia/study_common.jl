CUDA.device!(0); CUDA.context(); CUDA.stream!(CUDA.legacy_stream())
struct MiniLaunch
    op::Int32; n::Int32; grid::Int32; threads::Int32
    alpha::Float64; beta::Float64
    x::Ptr{Float64}; y::Ptr{Float64}; w::Ptr{Float64}
    row_offsets::Ptr{Int32}; cols::Ptr{Int32}; coefs::Ptr{Float64}
end
@assert sizeof(MiniLaunch)==80
const MINI_NNZ=Dict{Int32,Int}()
const MINI_COUNTS=zeros(Int,11)
@inline mini_wrap(p::Ptr{T},n) where T=unsafe_wrap(CuArray,CuPtr{T}(UInt(p)),(Int(n),);own=false)
function mini_callback(p::Ptr{MiniLaunch})::Cint
    try
        q=unsafe_load(p); @assert q.threads==256 && 0<=q.op<=10
        n=q.n; x=mini_wrap(q.x,n); w=mini_wrap(q.w,q.op==8 ? 256 : n)
        if q.op<8
            y=q.y==C_NULL ? x : mini_wrap(q.y,n)
            mini_vector_launch(x,y,w,n,q.alpha,q.beta,Int(q.grid),Val(Int(q.op)))
        elseif q.op==8
            y=mini_wrap(q.y,n)
            mini_dot_launch(x,y,w,n,Int(q.grid))
        elseif q.op==9
            @assert n==256 && q.grid==1
            mini_final_launch(w)
        else
            nz=MINI_NNZ[n]
            rp=mini_wrap(q.row_offsets,n+1); cols=mini_wrap(q.cols,nz); a=mini_wrap(q.coefs,nz)
            mini_matvec_launch(rp,cols,a,x,w,n,Int(q.grid))
        end
        MINI_COUNTS[Int(q.op)+1]+=1
        return Cint(0)
    catch err
        showerror(stderr,err,catch_backtrace());println(stderr);flush(stderr)
        return Cint(1)
    end
end
const MINI_CALLBACK=@cfunction(mini_callback,Cint,(Ptr{MiniLaunch},))
function mini_study()
    # The frozen ABI has no nnz field. Read actual retained native headers rather
    # than constructing oversized borrowed arrays outside the native allocations.
    for dim in (2,16,128)
        open(joinpath(ARGS[3],"actual_input_$(dim).bin")) do f
            magic=read(f,UInt64);d=read(f,UInt64);n=read(f,UInt64);nz=read(f,UInt64)
            @assert magic==0x4d4645494e505554 && d==dim && n==(dim+1)^3
            MINI_NNZ[Int32(n)]=Int(nz)
        end
    end
    path=abspath(ARGS[1]);lib=Libdl.dlopen(path);entry=Libdl.dlsym(lib,:mini_run)
    println("STUDY_RUNTIME_BINDING wrapper_sha256=",bytes2hex(sha256(read(path)))," original_serial_CPU_Q1_controller=true native_explicit_kernel_fallback=false")
    status=ccall(entry,Cint,(Cstring,Cstring,Cstring,Ptr{Cvoid},Cint),abspath(ARGS[2]),abspath(ARGS[3]),abspath(ARGS[4]),MINI_CALLBACK,0)
    CUDA.device_synchronize()
    println("STUDY_JULIA_ACTUAL_DISPATCH_COUNTS ",join(["op$(i-1)=$(MINI_COUNTS[i])" for i in eachindex(MINI_COUNTS)]," "))
    status==0 || error("Conditional full miniFE study failed status=",status)
end
