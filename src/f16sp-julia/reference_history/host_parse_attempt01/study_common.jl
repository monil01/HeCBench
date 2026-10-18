using CUDA,Printf,Libdl,SHA
const F16ROOT=@__DIR__
const F16VENDOR=joinpath(F16ROOT,"../../.porting-state/portable_julia/builds/f16sp/libstudy_vendor.so")
const F16LIB=Libdl.dlopen(F16VENDOR)
const F16CREATE=Libdl.dlsym(F16LIB,:study_create);const F16DESTROY=Libdl.dlsym(F16LIB,:study_destroy)
const F16BLAS=Libdl.dlsym(F16LIB,:study_cublas);const F16CUB=Libdl.dlsym(F16LIB,:study_cub)
function f16sp_study(factory)
    CUDA.device_synchronize(blocking=true);handle=Ref{Ptr{Cvoid}}(C_NULL);ccall(F16CREATE,Cint,(Ref{Ptr{Cvoid}},),handle)==0||error("cuBLAScreate")
    version=ccall(Libdl.dlsym(F16LIB,:study_cublas_version),Cint,(Ptr{Cvoid},),handle[])
    libpath=unsafe_string(ccall(Libdl.dlsym(F16LIB,:study_cublas_path),Cstring,()))
    println("VENDOR_PROVENANCE {\"cublas_version\":",version,",\"cublas_path\":\"",realpath(libpath),"\",\"cublas_sha256\":\"",bytes2hex(sha256(read(realpath(libpath)))),"\",\"bridge_sha256\":\"",bytes2hex(sha256(read(F16VENDOR))),"\"}")
    prepared=map((1,2)) do case
        fixture=collect(reinterpret(UInt16,read(joinpath(F16ROOT,"study_case$(case)_u16.bin"))));oracle=collect(reinterpret(Float64,read(joinpath(F16ROOT,"oracle_case$(case)_f64.bin")))))
        ba=CUDA.fill(reinterpret(Float16,UInt16(0x7bff)),65536+64);bb=similar(ba);fill!(bb,reinterpret(Float16,UInt16(0x7bff)));br=CUDA.fill(-8192f0,33);bh=CUDA.fill(reinterpret(Float16,UInt16(0x7bff)),65)
        a=view(ba,33:65568);b=view(bb,33:65568);out=view(br,17:17);hout=view(bh,33:33)
        copyto!(a,collect(reinterpret(Float16,fixture[1:65536])));copyto!(b,collect(reinterpret(Float16,fixture[65537:end])));run=factory(a,b,out,Int32(32768))
        function execute(mode,grid)
            if mode<=3
                CUDA.memset(reinterpret(CUDA.CuPtr{UInt8},pointer(out)),UInt8(0),4);run(mode,grid)
            elseif mode==4
                status=ccall(F16BLAS,Cint,(Ptr{Cvoid},CUDA.CuPtr{Cvoid},CUDA.CuPtr{Cvoid},CUDA.CuPtr{Cvoid},Int64),handle[],pointer(a),pointer(b),pointer(hout),65536);status==0||error("cuBLASDotEx64 status$(status)")
            else
                status=ccall(F16CUB,Cint,(CUDA.CuPtr{Cvoid},CUDA.CuPtr{Cvoid},CUDA.CuPtr{Cvoid},Int64),pointer(a),pointer(b),pointer(out),32768);status==0||error("CUBDeviceReduce status$(status)")
            end
            nothing
        end
        function guards()
            g=Float64[]
            for base in (ba,bb,bh)
                bits=collect(reinterpret(UInt16,Array(base)));append!(g,bits[1:32]);append!(g,bits[end-31:end]);all(==(UInt16(0x7bff)),bits[1:32])&&all(==(UInt16(0x7bff)),bits[end-31:end])||error("Halfguard")
            end
            raw=Array(br);append!(g,raw[1:16]);append!(g,raw[end-15:end]);all(==(-8192f0),raw[1:16])&&all(==(-8192f0),raw[end-15:end])||error("Floatguard")
            g
        end
        cases=[(mode,grid) for grid in (256,128,64,32,16) for mode in 1:3];append!(cases,[(4,0),(5,0)])
        for (mode,grid) in cases;execute(mode,grid);CUDA.device_synchronize(blocking=true);end
        (;case,fixture,oracle,a,b,out,hout,execute,guards,cases)
    end
    allok=true
    for trial in 1:5
        prefix=Float64[];outputs=Float64[];guardvalues=Float64[];times=Float64[];ma=0.;mr=0.;scaled=0.;ok=true
        for z in prepared
            initial=vcat(collect(reinterpret(UInt16,Array(z.a))),collect(reinterpret(UInt16,Array(z.b))));ok&=initial==z.fixture;append!(prefix,initial);append!(prefix,Float64[z.case,32768,65536,3,128,256,128,64,32,16,17,1])
            for (k,(mode,grid)) in enumerate(z.cases)
                CUDA.device_synchronize(blocking=true);start=time_ns();for _ in 1:3;z.execute(mode,grid);end;CUDA.device_synchronize(blocking=true);push!(times,(time_ns()-start)*1e-9/3)
                value=mode==4 ? Float64(Array(z.hout)[1]) : Float64(Array(z.out)[1]);push!(outputs,value);append!(guardvalues,z.guards());e=abs(value-z.oracle[k]);ma=max(ma,e);mr=max(mr,e/max(abs(z.oracle[k]),eps(Float64)));scaled=max(scaled,e/(.02+2e-6*abs(z.oracle[k])));ok&=isfinite(value)&&e<=.02+2e-6*abs(z.oracle[k]);if mode==4;ok&=value==z.oracle[k];end
            end
            ok&=vcat(collect(reinterpret(UInt16,Array(z.a))),collect(reinterpret(UInt16,Array(z.b))))==z.fixture
        end
        allok&=ok;artifact=vcat(prefix,outputs,guardvalues);path=joinpath(ENV["STUDY_OUTPUT_DIR"],"trial$(trial).bin");write(path,artifact)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":7650,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"relative_error_floor\":2.220446049250313e-16,\"max_scaled_oracle_error\":%.12g,\"timing_region\":\"sum_two_fixtures_all17_mixed_precision_custom_reset_and_vendor_API_means\",\"jit_excluded\":true,\"internal_repeats\":3,\"timed_cases\":34,\"component_time_seconds\":[",trial,sum(times),ok ? "true" : "false",ma,mr,scaled)
        for i in eachindex(times);i>1&&print(",");@printf("%.12g",times[i]);end
        @printf("],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",path,length(artifact))
    end
    ccall(F16DESTROY,Cint,(Ptr{Cvoid},),handle[])==0||error("cuBLASdestroy");allok||error("Distinctmixedprecisionoutputs/oracles mismatch");println("PASS")
end
