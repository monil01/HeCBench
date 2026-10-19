using CUDA,Printf
include("../filter_device.jl")
function kernel_1!(imgNDVI,imgQA,n_X,n_Y,n_B,n_Years,img_NDVI,img_QA)
 ids=(UInt32(blockIdx().x-1),UInt32(blockIdx().y-1),UInt32(blockDim().x),UInt32(blockDim().y),UInt32(threadIdx().x-1),UInt32(threadIdx().y-1))
 Short_to_Float_body!(imgNDVI,imgQA,n_X,n_Y,n_B,n_Years,img_NDVI,img_QA,ids...)
 return
end
function kernel_2!(cosyear,win_NDVI,img_NDVI,img_QA,n_X,n_Y,n_B,n_Years,reference_data,d_res_3,d_res_vec_res1)
 ids=(UInt32(blockIdx().x-1),UInt32(blockIdx().y-1),UInt32(blockDim().x),UInt32(blockDim().y),UInt32(threadIdx().x-1),UInt32(threadIdx().y-1))
 Generate_NDVI_reference_body!(cosyear,win_NDVI,img_NDVI,img_QA,n_X,n_Y,n_B,n_Years,reference_data,d_res_3,d_res_vec_res1,ids...)
 return
end
function kernel_3!(img_NDVI,img_QA,reference_data,StartY,TotalY,Buffer_Up,Buffer_Dn,n_X,n_Y,n_B,n_Years,win,d_res)
 ids=(UInt32(blockIdx().x-1),UInt32(blockIdx().y-1),UInt32(blockDim().x),UInt32(blockDim().y),UInt32(threadIdx().x-1),UInt32(threadIdx().y-1))
 Compute_d_res_body!(img_NDVI,img_QA,reference_data,StartY,TotalY,Buffer_Up,Buffer_Dn,n_X,n_Y,n_B,n_Years,win,d_res,ids...)
 return
end
function kernel_4!(img_NDVI,img_QA,reference_data,StartY,TotalY,Buffer_Up,Buffer_Dn,n_X,n_Y,n_B,n_Years,win,sampcorr,snow_address,vector_out,d_vector_in,d_res,d_res_3,d_index)
 ids=(UInt32(blockIdx().x-1),UInt32(blockIdx().y-1),UInt32(blockDim().x),UInt32(blockDim().y),UInt32(threadIdx().x-1),UInt32(threadIdx().y-1))
 STSG_filter_body!(img_NDVI,img_QA,reference_data,StartY,TotalY,Buffer_Up,Buffer_Dn,n_X,n_Y,n_B,n_Years,win,sampcorr,snow_address,vector_out,d_vector_in,d_res,d_res_3,d_index,ids...)
 return
end
function factory(X,Y)
 return function(stage,args...)
  if stage==1
   @cuda threads=(16,16) blocks=(cld(X,16),cld(Y,16)) kernel_1!(args...)
  elseif stage==2
   @cuda threads=(16,16) blocks=(cld(X,16),cld(Y,16)) kernel_2!(args...)
  elseif stage==3
   @cuda threads=(16,16) blocks=(cld(X,16),cld(Y,16)) kernel_3!(args...)
  elseif stage==4
   @cuda threads=(16,16) blocks=(cld(X,16),cld(Y,16)) kernel_4!(args...)
  end
  nothing
 end
end
include("../study_common.jl")
stsg_study(factory)
