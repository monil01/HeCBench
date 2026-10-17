using CUDA
using Printf
const STUDY_TYPES=(
    (name="uchar_misaligned",T=UInt8,packed=1,stride=1,alignment=1),
    (name="uchar4_misaligned",T=NTuple{4,UInt8},packed=4,stride=4,alignment=1),
    (name="ushort_misaligned",T=UInt16,packed=2,stride=2,alignment=2),
    (name="uint2_misaligned",T=NTuple{2,UInt32},packed=8,stride=8,alignment=4),
    (name="uint3_misaligned",T=NTuple{3,UInt32},packed=12,stride=12,alignment=4),
    (name="uint4_misaligned",T=NTuple{4,UInt32},packed=16,stride=16,alignment=4),
    (name="uint8_misaligned",T=NTuple{8,UInt32},packed=32,stride=32,alignment=4),
    (name="uchar4_aligned",T=UInt32,packed=4,stride=4,alignment=4),
    (name="uint_aligned",T=UInt32,packed=4,stride=4,alignment=4),
    (name="uint2_aligned",T=UInt64,packed=8,stride=8,alignment=8),
    (name="uint3_aligned",T=UInt128,packed=12,stride=16,alignment=16),
    (name="uint4_aligned",T=UInt128,packed=16,stride=16,alignment=16),
    (name="uint8_aligned",T=NTuple{2,UInt128},packed=32,stride=32,alignment=16),
)

function study_loop!(launch!,repeat)
    for _ in 1:repeat
        launch!()
    end
end
function study_main(prepare,args=ARGS)
    bytes=196608;repeat=isempty(args) ? 100 : parse(Int,args[1]);repeat>0||error("positive repeat");input=UInt8[UInt8((i*37+(i÷256)*13+17)%256) for i in 0:bytes-1];src=CuArray(input);dst=CUDA.zeros(UInt8,bytes);fixtures=[]
    for spec in STUDY_TYPES
        sizeof(spec.T)==spec.stride||error("stride mismatch");Base.datatype_alignment(spec.T)==spec.alignment||error("ABI alignment mismatch")
        a=reinterpret(spec.T,src);b=reinterpret(spec.T,dst);n=length(a);launch! = prepare(b,a,n);study_loop!(launch!,1);push!(fixtures,(spec,launch!,n))
        @printf("STUDY_LAYOUT {\"name\":\"%s\",\"sizeof\":%d,\"alignment\":%d,\"packed_bytes\":%d}\n",spec.name,sizeof(spec.T),Base.datatype_alignment(spec.T),spec.packed)
    end
    CUDA.synchronize();allok=true
    for trial in 1:5
        output=UInt8[];times=Float64[];ok=true;ma=0;gct=0.0;gcp=0
        for (spec,launch!,n) in fixtures
            CUDA.fill!(dst,UInt8(0));CUDA.synchronize();gb=Base.gc_num();t0=time_ns();study_loop!(launch!,repeat);CUDA.synchronize();elapsed=(time_ns()-t0)*1e-9/repeat;ga=Base.gc_num();push!(times,elapsed);gct+=(ga.total_time-gb.total_time)*1e-9;gcp+=ga.pause-gb.pause;got=Array(dst)
            for e in 0:n-1,j in 1:spec.packed
                idx=e*spec.stride+j;diff=abs(Int(got[idx])-Int(input[idx]));ma=max(ma,diff);ok &=diff==0;push!(output,got[idx])
            end
        end
        allok &=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"aligned_trial$(trial).bin");write(file,output)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%d,\"max_rel_error\":0,\"timing_region\":\"sum_thirteen_ABI_layout_copy_regions_per_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"layout_times_seconds\":[%s],\"gc_time_seconds_whole_region\":%.12g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint8\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),string(ok),length(output),ma,repeat,join(times,","),gct,gcp,file,length(output))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
