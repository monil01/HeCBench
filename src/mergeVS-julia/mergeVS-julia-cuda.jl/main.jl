using CUDA
function metadata_study!(q,kv,vertical,slash,vc,sc,bc,bo,cc,ci,H,R,M,BN,V,S,causal)
 h=Int64(blockIdx().x-1);b=Int64(blockIdx().y-1);row=Int64(blockIdx().z-1)*64+Int64(threadIdx().x-1)
 metadata_row!(row,h,b,q,kv,vertical,slash,vc,sc,bc,bo,cc,ci,H,R,M,BN,V,S,causal)
 return
end
