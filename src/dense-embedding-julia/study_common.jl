using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function reference(input, dense, embedding_dim, batch_size, offset)
    output = zeros(Float32, length(input))
    for batch_idx in 0:(batch_size - 1)
        range = offset[batch_idx + 2] - offset[batch_idx + 1]
        for idx in 0:(embedding_dim - 1)
            dense_elem = dense[batch_idx * embedding_dim + idx + 1]
            nested_idx = idx
            while nested_idx < range
                out_idx = offset[batch_idx + 1] + nested_idx + 1
                output[out_idx] = input[out_idx] + dense_elem
                nested_idx += embedding_dim
            end
        end
    end
    return output
end

function dense_study(factory)
    mode=length(ARGS)>0 ? ARGS[1] : "k1";rows=512;batch=16;dim=768;repeats=10
    input=Float32[Float32(mod(17*i,257)-128)/128f0 for i in 0:rows*dim-1]
    dense=Float32[Float32(mod(13*i,127)-63)/64f0 for i in 0:batch*dim-1]
    offsets=zeros(Int32,batch+1)
    for i in 0:batch-1;offsets[i+2]=offsets[i+1]+Int32((2+mod(i,7))*dim);end
    expected=reference(input,dense,dim,batch,offsets)
    di=CuArray(input);dd=CuArray(dense);doff=CuArray(offsets);output=CUDA.zeros(Float32,length(input));launch! = factory(di,dd,output,doff,mode)
    OwnedStudy.run_trials!(launch!,()->fill!(output,0f0),()->Array(output),expected,repeats;atol=1e-3,rtol=0.)
end
