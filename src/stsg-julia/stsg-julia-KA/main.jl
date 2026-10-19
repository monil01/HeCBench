using CUDA,KernelAbstractions,Printf
include("../filter_device.jl")
@kernel function kernel_1!(imgNDVI,imgQA,n_X,n_Y,n_B,n_Years,img_NDVI,img_QA)
 group=@index(Group,NTuple);localid=@index(Local,NTuple)
 ids=(UInt32(group[1]-1),UInt32(group[2]-1),UInt32(16),UInt32(16),UInt32(localid[1]-1),UInt32(localid[2]-1))
 Short_to_Float_body!(imgNDVI,imgQA,n_X,n_Y,n_B,n_Years,img_NDVI,img_QA,ids...)
end
@kernel function kernel_2!(cosyear,win_NDVI,img_NDVI,img_QA,n_X,n_Y,n_B,n_Years,reference_data,d_res_3,d_res_vec_res1)
 group=@index(Group,NTuple);localid=@index(Local,NTuple)
 ids=(UInt32(group[1]-1),UInt32(group[2]-1),UInt32(16),UInt32(16),UInt32(localid[1]-1),UInt32(localid[2]-1))
 Generate_NDVI_reference_body!(cosyear,win_NDVI,img_NDVI,img_QA,n_X,n_Y,n_B,n_Years,reference_data,d_res_3,d_res_vec_res1,ids...)
end
@kernel function kernel_3!(img_NDVI,img_QA,reference_data,StartY,TotalY,Buffer_Up,Buffer_Dn,n_X,n_Y,n_B,n_Years,win,d_res)
 group=@index(Group,NTuple);localid=@index(Local,NTuple)
 ids=(UInt32(group[1]-1),UInt32(group[2]-1),UInt32(16),UInt32(16),UInt32(localid[1]-1),UInt32(localid[2]-1))
 Compute_d_res_body!(img_NDVI,img_QA,reference_data,StartY,TotalY,Buffer_Up,Buffer_Dn,n_X,n_Y,n_B,n_Years,win,d_res,ids...)
end
@kernel function kernel_4!(img_NDVI,img_QA,reference_data,StartY,TotalY,Buffer_Up,Buffer_Dn,n_X,n_Y,n_B,n_Years,win,sampcorr,snow_address,vector_out,d_vector_in,d_res,d_res_3,d_index)
 group=@index(Group,NTuple);localid=@index(Local,NTuple)
 ids=(UInt32(group[1]-1),UInt32(group[2]-1),UInt32(16),UInt32(16),UInt32(localid[1]-1),UInt32(localid[2]-1))
 STSG_filter_body!(img_NDVI,img_QA,reference_data,StartY,TotalY,Buffer_Up,Buffer_Dn,n_X,n_Y,n_B,n_Years,win,sampcorr,snow_address,vector_out,d_vector_in,d_res,d_res_3,d_index,ids...)
end
function factory(X,Y)
 backend=CUDA.CUDABackend()
 kernels=(kernel_1!(backend,(16,16)),kernel_2!(backend,(16,16)),kernel_3!(backend,(16,16)),kernel_4!(backend,(16,16)))
 return (stage,args...)->kernels[stage](args...;ndrange=(cld(X,16)*16,cld(Y,16)*16))
end
include("../study_common.jl")
stsg_study(factory)
