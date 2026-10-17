include(joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function study_vendor!(net)
 CUDA.synchronize();st=time_ns()
 di=CuArray(net.input_units[2:end]);dw=CuArray(net.input_weights[2:end,2:end]);db=CuArray(net.input_weights[1,2:end]);dh=sigmoid.(transpose(dw)*di .+db)
 net.hidden_units[1]=1f0;net.hidden_units[2:end].=Array(dh)
 layerforward!(net.hidden_units,net.output_units,net.hidden_weights,16,1);eo=output_error!(net.output_delta,net.target,net.output_units,1);eh=hidden_error!(net.hidden_delta,16,net.output_delta,1,net.hidden_weights,net.hidden_units)
 adjust_weights!(net.output_delta,1,net.hidden_units,16,net.hidden_weights,net.hidden_prev_weights)
 dd=CuArray(net.hidden_delta[2:end]);dp=CuArray(net.input_prev_weights[2:end,2:end]);dbp=CuArray(net.input_prev_weights[1,2:end]);du=ETA.*(di*transpose(dd)).+MOMENTUM.*dp;dbu=ETA.*dd.+MOMENTUM.*dbp;dw.+=du;db.+=dbu
 w=copy(net.input_weights);prev=copy(net.input_prev_weights);w[2:end,2:end].=Array(dw);w[1,2:end].=Array(db);prev[2:end,2:end].=Array(du);prev[1,2:end].=Array(dbu)
 for a in (di,dw,db,dh,dd,dp,dbp,du,dbu);CUDA.unsafe_free!(a);end;CUDA.synchronize();elapsed=(time_ns()-st)*1e-9
 pack_result(net,rowflat(w),rowflat(prev),eo,eh),elapsed
end
study_main(study_vendor!)
