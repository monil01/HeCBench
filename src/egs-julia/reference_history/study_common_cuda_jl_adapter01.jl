using CUDA, Libdl, Printf
CUDA.device!(0)
CUDA.context()
const EGS_HOST_LIB=Libdl.dlopen(abspath(ARGS[1]))
const EGS_DATA=abspath(ARGS[2])
const EGS_ROOT=@__DIR__
const EGS_CREATE=Libdl.dlsym(EGS_HOST_LIB,:egs_create)
const EGS_NAME=Libdl.dlsym(EGS_HOST_LIB,:egs_name)
const EGS_ENCODING=Libdl.dlsym(EGS_HOST_LIB,:egs_encoding)
const EGS_BYTES=Libdl.dlsym(EGS_HOST_LIB,:egs_bytes)
const EGS_MUTABLE=Libdl.dlsym(EGS_HOST_LIB,:egs_mutable)
const EGS_POINTER=Libdl.dlsym(EGS_HOST_LIB,:egs_pointer)
const EGS_COPY=Libdl.dlsym(EGS_HOST_LIB,:egs_copy)
const EGS_GUARDS=Libdl.dlsym(EGS_HOST_LIB,:egs_guards)
const EGS_READONLY=Libdl.dlsym(EGS_HOST_LIB,:egs_readonly)
const EGS_SCALAR=Libdl.dlsym(EGS_HOST_LIB,:egs_scalar)
const EGS_RESET=Libdl.dlsym(EGS_HOST_LIB,:egs_reset)
const EGS_DESTROY=Libdl.dlsym(EGS_HOST_LIB,:egs_destroy)
# There is deliberately no lookup of native GPU simulation/accumulation entrypoints.
function egs_prepare()
    n=ccall(EGS_CREATE,Cint,(Cstring,),EGS_DATA)
    descriptors=NamedTuple[]; arrays=Dict{String,Any}()
    for i in 0:n-1
        name=unsafe_string(ccall(EGS_NAME,Cstring,(Cint,),i))
        enc=unsafe_string(ccall(EGS_ENCODING,Cstring,(Cint,),i))
        bytes=Int(ccall(EGS_BYTES,Csize_t,(Cint,),i))
        mutable=ccall(EGS_MUTABLE,Cint,(Cint,),i)!=0
        ptr=ccall(EGS_POINTER,Ptr{Cvoid},(Cint,),i)
        T=enc=="uint32" ? UInt32 : enc=="int32" ? Int32 : enc=="uint64" ? UInt64 : enc=="float32" ? Float32 : Float64
        arrays[name]=unsafe_wrap(CuArray,CuPtr{T}(ptr),(bytes÷sizeof(T),);own=false)
        push!(descriptors,(index=i,name=name,encoding=enc,bytes=bytes,mutable=mutable))
    end
    names=(:mt_tables,:mt_params,:mt_state,:stack_a,:stack_b,:stack_c,:step_counts,:list_depth,
           :inner_iterations,:total_weight,:source_edges,:source_weights,:source_alias,:x_bounds,
           :y_bounds,:z_bounds,:regions,:ge,:gmfp,:gbr1,:gbr2,:cohe,:pmax,:rayleigh,:rayleigh_indices)
    context=NamedTuple{names}(Tuple(arrays[String(k)] for k in names))
    context=merge(context,(sc_count=ntuple(k->arrays["score_count_$(k-1)"],4),
        sc_energy=ntuple(k->arrays["score_energy_$(k-1)"],4),
        det_count=ntuple(k->arrays["total_count_$(k-1)"],4),
        det_energy=ntuple(k->arrays["total_energy_$(k-1)"],4)))
    scalars=[ccall(EGS_SCALAR,Cdouble,(Cint,),i) for i in 0:21]
    return context,descriptors,scalars
end
function egs_capture(directory,label,descriptors,full)
    records=String[]
    for d in descriptors
        (!full && !d.mutable) && continue
        raw=Vector{UInt8}(undef,d.bytes)
        ccall(EGS_COPY,Cvoid,(Cint,Ptr{UInt8}),d.index,raw)
        name="$(label)_$(d.name).bin"; write(joinpath(directory,name),raw)
        push!(records,"\"$(d.name)\":{\"path\":\"$name\",\"bytes\":$(d.bytes)}")
    end
    guards=ccall(EGS_GUARDS,Cint,())==1
    readonly=ccall(EGS_READONLY,Cint,())==1
    @assert guards && readonly
    return "{\"label\":\"$label\",\"records\":{$(join(records,","))},\"guards_pass\":true,\"all_readonly_unchanged\":true}"
end
function egs_study()
    context,descriptors,scalars=egs_prepare()
    # Whole-context fences are required for Julia/CUDA-runtime pointer interoperation.
    for phase in 0:7
        launch_simulation(context,phase==0,phase!=0)
        launch_accumulation(context)
        CUDA.device_synchronize()
        counts=Array(context.step_counts)
        sum(counts[1:8])==0 && break
        @assert phase<7
    end
    for trial in 1:5
        ccall(EGS_RESET,Cvoid,()); CUDA.device_synchronize()
        directory="capture_trial$trial"; mkdir(directory)
        captures=String[egs_capture(directory,"initial",descriptors,true)]
        segments=Float64[]
        for phase in 0:7
            CUDA.device_synchronize(); t=time_ns()
            launch_simulation(context,phase==0,phase!=0)
            CUDA.device_synchronize(); t1=time_ns()
            launch_accumulation(context)
            CUDA.device_synchronize(); t2=time_ns()
            append!(segments,[(t1-t)*1e-9,(t2-t1)*1e-9])
            push!(captures,egs_capture(directory,"phase$phase",descriptors,false))
            counts=Array(context.step_counts)
            sum(counts[1:8])==0 && break
            @assert phase<7
        end
        ds=["{\"index\":$(d.index),\"name\":\"$(d.name)\",\"encoding\":\"$(d.encoding)\",\"bytes\":$(d.bytes),\"mutable\":$(d.mutable)}" for d in descriptors]
        text="{\"role\":\"Julia_checked_trial\",\"scalars\":[$(join(scalars,","))],\"allocations\":[$(join(ds,","))],\"captures\":[$(join(captures,","))],\"segment_seconds\":[$(join(segments,","))]}"
        write(joinpath(directory,"capture_manifest.json"),text)
        run(`python3 $(joinpath(EGS_ROOT,"finalize_capture.py")) --capture-dir $directory --trial $trial`)
    end
    CUDA.device_synchronize(); ccall(EGS_DESTROY,Cvoid,())
end
