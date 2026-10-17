using CUDA
using Printf
@inline function study_sigmoid_f32(x::Float32)
    return 1.0f0 / (1.0f0 + exp(-x))
end

function study_reference(input, hidden, bias1, bias2, hx, hsz::Int, total_elements::Int)
    hy = Vector{Float16}(undef, total_elements)
    storage = Vector{Float16}(undef, 5 * total_elements)
    @inbounds for linear in 0:total_elements-1
        batch_idx = linear ÷ hsz
        hidden_idx = linear % hsz
        offset = batch_idx * 3 * hsz + hidden_idx + 1
        bidx = hidden_idx + 1

        rg = study_sigmoid_f32(Float32(input[offset]) + Float32(hidden[offset]) +
                         Float32(bias1[bidx]) + Float32(bias2[bidx]))
        ig = study_sigmoid_f32(Float32(input[offset + hsz]) + Float32(hidden[offset + hsz]) +
                         Float32(bias1[bidx + hsz]) + Float32(bias2[bidx + hsz]))
        ng = tanh(Float32(input[offset + 2 * hsz]) + Float32(bias1[bidx + 2 * hsz]) +
                  rg * (Float32(hidden[offset + 2 * hsz]) + Float32(bias2[bidx + 2 * hsz])))
        hy[linear + 1] = Float16(ng + ig * (Float32(hx[linear + 1]) - ng))

        store_offset = batch_idx * 5 * hsz + hidden_idx + 1
        storage[store_offset] = Float16(rg)
        storage[store_offset + hsz] = Float16(ig)
        storage[store_offset + 2 * hsz] = Float16(ng)
        storage[store_offset + 3 * hsz] = hx[linear + 1]
        storage[store_offset + 4 * hsz] = Float16(Float32(hidden[offset + 2 * hsz]) +
                                                  Float32(bias2[bidx + 2 * hsz]))
    end
    return hy, storage
end


function study_loop!(launch!,repeat)
    for _ in 1:repeat
        launch!()
    end
end
function study_fixture(n,mul,add)
    Float16[Float16(Float32((i*mul+add)%257-128)/128f0) for i in 0:n-1]
end
function study_main(prepare,args=ARGS)
    h=256;batch=64;n=h*batch;repeat=isempty(args) ? 100 : parse(Int,args[1]);repeat>0||error("positive repeat")
    input=study_fixture(3*n,13,7);hidden=study_fixture(3*n,37,11);b1=study_fixture(3*h,17,3);b2=study_fixture(3*h,29,5);hx=study_fixture(n,43,19);hyref,sref=study_reference(input,hidden,b1,b2,hx,h,n);expected=vcat(Float32.(hyref),Float32.(sref))
    di,dh,db1,db2,dx=CuArray(input),CuArray(hidden),CuArray(b1),CuArray(b2),CuArray(hx);hy=CUDA.zeros(Float16,n);storage=CUDA.zeros(Float16,5*n);launch! = prepare(di,dh,db1,db2,dx,hy,storage,h,n);study_loop!(launch!,1);CUDA.synchronize();allok=true
    for trial in 1:5
        CUDA.synchronize();gb=Base.gc_num();t0=time_ns();study_loop!(launch!,repeat);CUDA.synchronize();elapsed=(time_ns()-t0)*1e-9/repeat;ga=Base.gc_num();got=vcat(Float32.(Array(hy)),Float32.(Array(storage)));err=abs.(got.-expected);ma=maximum(err);mr=maximum(err./max.(abs.(expected),eps(Float32)));ok=all(isfinite,got)&&ma<=1f-3;allok &=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"gru_trial$(trial).bin");write(file,got)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"gru_cell_forward_including_all_five_saved_states_per_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"gc_time_seconds_whole_region\":%.12g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,string(ok),length(got),ma,mr,repeat,(ga.total_time-gb.total_time)*1e-9,ga.pause-gb.pause,file,length(got))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
