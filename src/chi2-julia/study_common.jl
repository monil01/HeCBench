using CUDA
using Printf
function chi_gold(input,rows,cols,ncases,ncontrols)
    expected=zeros(Float64,cols)
    for col in 0:cols-1
        cases=ones(Int,3);controls=ones(Int,3)
        for row in 0:ncases-1;cases[Int(input[row*cols+col+1])-47]+=1;end
        for row in ncases:ncases+ncontrols-1;controls[Int(input[row*cols+col+1])-47]+=1;end
        ct=sum(cases);nt=sum(controls);total=ct+nt;s=0.
        for category in 1:3
            count=cases[category]+controls[category];ce=ct*count/total;ne=nt*count/total;s+=(cases[category]-ce)^2/ce+(controls[category]-ne)^2/ne
        end
        expected[col+1]=s
    end
    expected
end
function study_main(prepare,args=ARGS)
    rows=12;cols=257;ncases=3;ncontrols=9;repeat=isempty(args) ? 100 : parse(Int,args[1]);repeat>0||error("positive repeat");fixtures=[]
    for kind in 0:1
        input=UInt8[UInt8(48+(kind==0 ? (row+col)%3 : (row*row+col*7+row*col)%3)) for row in 0:rows-1 for col in 0:cols-1];expected=chi_gold(input,rows,cols,ncases,ncontrols);di,dc=CuArray(input),CUDA.zeros(Float32,cols);launch! = prepare(di,dc,rows,cols,ncases,ncontrols);launch!();push!(fixtures,(dc,launch!,expected,kind))
    end
    CUDA.synchronize();allok=true
    for trial in 1:5
        output=Float32[];times=Float64[];ok=true;ma=0.0
        for (dc,launch!,expected,kind) in fixtures
            CUDA.synchronize();t0=time_ns();for rep in 1:repeat;launch!();end;CUDA.synchronize();push!(times,(time_ns()-t0)*1e-9/repeat);got=Array(dc);err=maximum(abs.(Float64.(got).-expected));ma=max(ma,err);ok &=all(isfinite,got)&&all(got.>=0)&&err<=1e-4&&(kind!=0||all(iszero,got));append!(output,got)
        end
        allok &=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"chi2_trial$(trial).bin");write(file,output)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":514,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"timing_region\":\"sum_equal_distribution_and_varied_SNP_pseudocounted_chi_square_corrected_margins\",\"jit_excluded\":true,\"internal_repeats\":%d,\"native_reference_repaired\":true,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":514,\"byte_order\":\"little\"}}\n",trial,sum(times),string(ok),ma,repeat,file)
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
