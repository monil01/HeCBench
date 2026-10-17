using CUDA
using Printf
function study_main(prepare,args=ARGS)
    repeat=isempty(args) ? 30 : parse(Int,args[1]);repeat>0||error("positive repeat");data=reinterpret(Float64,read(joinpath(@__DIR__,"study_fixture.bin")));length(data)==123264||error("fixture length")
    vgeo=copy(data[1:12288]);cg=copy(data[12289:110592]);D=copy(data[110593:110848]);I=copy(data[110849:110976]);prefix=copy(data[1:117120]);gold=copy(data[117121:end]);dv,dc,dd,di=CuArray(vgeo),CuArray(cg),CuArray(D),CuArray(I);fixtures=[]
    for kind in 0:1
        u=copy(data[110977+kind*3072:114048+kind*3072]);out=CUDA.zeros(Float64,3072);input=CuArray(u);launch! = prepare(dv,dc,dd,di,input,out,2);launch!();push!(fixtures,(out,launch!,copy(gold[1+kind*3072:3072+kind*3072])))
    end
    CUDA.synchronize();allok=true
    for trial in 1:5
        output=copy(prefix);seconds=0.0;ma=0.0;ok=true
        for (out,launch!,expected) in fixtures
            CUDA.synchronize();t0=time_ns();for rep in 1:repeat;launch!();end;CUDA.synchronize();seconds+=(time_ns()-t0)*1e-9/repeat;got=Array(out);err=maximum(abs.(got.-expected));ma=max(ma,err);ok &=all(isfinite,got)&&all(abs.(got.-expected).<=1e-7.+1e-11.*abs.(expected));append!(output,got)
        end
        allok &=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"adv_trial$(trial).bin");write(file,output)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"computed_output_values\":6144,\"input_prefix_values\":117120,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"timing_region\":\"sum_two_full_source_corner_3D_cubature_advection_operators\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,string(ok),length(output),ma,repeat,file,length(output))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
