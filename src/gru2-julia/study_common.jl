function gru!(run_layer!,d_input, d_w_ih, d_w_hh, d_b_ih, d_b_hh, d_h0, d_output,
              d_hidden, d_hidden_a, d_layer_output_a, d_layer_output_b,
              seq_len, batch_size, input_size, hidden_size, num_layers,
              max_input_size)
    hidden_read = d_h0
    hidden_buffers = (d_hidden, d_hidden_a)
    hidden_write_idx = 1
    blocks = cld(batch_size * hidden_size, THREADS)

    for t in 0:(seq_len - 1)
        hidden_write = hidden_buffers[hidden_write_idx]
        current_input = @view d_input[(t * batch_size * input_size + 1):((t + 1) * batch_size * input_size)]
        current_input_size = input_size
        output_t = @view d_output[(t * batch_size * hidden_size + 1):((t + 1) * batch_size * hidden_size)]

        for layer in 0:(num_layers - 1)
            hidden_offset = layer * batch_size * hidden_size
            layer_hidden_read = @view hidden_read[(hidden_offset + 1):(hidden_offset + batch_size * hidden_size)]
            layer_hidden_write = @view hidden_write[(hidden_offset + 1):(hidden_offset + batch_size * hidden_size)]
            current_output = iseven(layer) ? d_layer_output_a : d_layer_output_b
            if layer > 0
                current_input = isodd(layer) ? d_layer_output_a : d_layer_output_b
                current_input_size = hidden_size
            end
            write_output = layer == num_layers - 1 ? Int32(1) : Int32(0)
            run_layer!(
                current_input, layer_hidden_read, d_w_ih, d_w_hh, d_b_ih, d_b_hh,
                current_output, layer_hidden_write, output_t, Int32(layer),
                Int32(batch_size), Int32(current_input_size), Int32(hidden_size),
                Int32(max_input_size), write_output)
        end
        hidden_read = hidden_write
        hidden_write_idx = 3 - hidden_write_idx
    end
    return hidden_read
end

const STUDY_CASES=((3,3,37,35,3,false),(4,2,5,7,2,true))
const STUDY_REPEATS=3
function fixture(S,B,I,H,L,zero)
 M=max(I,H);sizes=(S*B*I,L*3*H*M,L*3*H*H,L*3*H,L*3*H,L*B*H)
 arrays=[Float32[Float32(mod(17i+13a,31)-15)/Float32(a in (2,3) ? 128 : 64) for i in 0:n-1] for (a,n) in enumerate(sizes)]
 if zero
  for a in 2:5;fill!(arrays[a],0);end
 else
  # Poison padding ignored by deeper-layer input size H, but keep live weights unchanged.
  for layer in 1:L-1,gate in 0:2,h in 0:H-1,k in H:M-1
   arrays[2][(((layer*3+gate)*H+h)*M+k)+1]=55.25f0
  end
 end
 arrays
end
function oracle(arrays,S,B,I,H,L)
 input,wi,wh,bi,bh,h0=arrays;M=max(I,H);hidden=Float64.(h0);output=Float64[]
 for t in 0:S-1
  current=Float64.(input[t*B*I+1:(t+1)*B*I]);size_in=I
  for layer in 0:L-1
   next=zeros(Float64,B*H)
   for b in 0:B-1,h in 0:H-1
    ig=[Float64(bi[(layer*3+g)*H+h+1])+sum(current[b*size_in+k+1]*Float64(wi[(((layer*3+g)*H+h)*M+k)+1]) for k in 0:size_in-1) for g in 0:2]
    hg=[Float64(bh[(layer*3+g)*H+h+1])+sum(hidden[(layer*B+b)*H+k+1]*Float64(wh[(((layer*3+g)*H+h)*H+k)+1]) for k in 0:H-1) for g in 0:2]
    reset=1/(1+exp(-(ig[1]+hg[1])));update=1/(1+exp(-(ig[2]+hg[2])));candidate=tanh(ig[3]+reset*hg[3]);hx=hidden[(layer*B+b)*H+h+1];next[b*H+h+1]=(1-update)*candidate+update*hx
   end
   hidden[layer*B*H+1:(layer+1)*B*H]=next;current=next;size_in=H
  end
  append!(output,current)
 end
 vcat(output,hidden)
end
function study(run_layer!)
 cases=[];prefix=Float32[]
 for(S,B,I,H,L,zero)in STUDY_CASES
  arrays=fixture(S,B,I,H,L,zero);for a in arrays;append!(prefix,a);end;append!(prefix,Float32[S,B,I,H,L])
  gpu=CuArray.(arrays);out=CUDA.zeros(Float32,S*B*H);ha=CUDA.zeros(Float32,L*B*H);hb=CUDA.zeros(Float32,L*B*H);la=CUDA.zeros(Float32,B*H);lb=CUDA.zeros(Float32,B*H);push!(cases,(gpu,out,ha,hb,la,lb,S,B,I,H,L,oracle(arrays,S,B,I,H,L)))
 end
 function case!(args;repeats=STUDY_REPEATS)
  gpu,out,ha,hb,la,lb,S,B,I,H,L,want=args;last=ha;CUDA.synchronize();t=time_ns()
  for _ in 1:repeats
   last=gru!(run_layer!,gpu[1],gpu[2],gpu[3],gpu[4],gpu[5],gpu[6],out,ha,hb,la,lb,S,B,I,H,L,max(I,H))
  end
  CUDA.synchronize();elapsed=(time_ns()-t)*1e-9/repeats;got=vcat(Array(out),Array(last));err=maximum(abs.(Float64.(got).-want));return got,elapsed,err
 end
 for args in cases;case!(args;repeats=1);end
 for trial in 1:5
  output=copy(prefix);elapsed=0.;err=0.
  for args in cases;got,t,e=case!(args);append!(output,got);elapsed+=t;err=max(err,e);end
  ok=all(isfinite,output)&&err<=1e-4;path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,output),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":25422,\"max_abs_error\":",err,",\"timing_region\":\"sum_two_multilayer_gru_forward_means\",\"jit_excluded\":true,\"internal_repeats\":3,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float32\",\"count\":25422,\"byte_order\":\"little\"}}")
  ok||Base.error("FullGRUsequence/hiddenoraclefailed")
 end
end
