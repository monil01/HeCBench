using CUDA,JACC,Printf
include("../filter_device.jl")
@inline function kernel_1!(i,j,imgNDVI,imgQA,n_X,n_Y,n_B,n_Years,img_NDVI,img_QA)
 ids=(UInt32(div(i-1,16)),UInt32(div(j-1,16)),UInt32(16),UInt32(16),UInt32(rem(i-1,16)),UInt32(rem(j-1,16)))
 Short_to_Float_body!(imgNDVI,imgQA,n_X,n_Y,n_B,n_Years,img_NDVI,img_QA,ids...)
 return nothing
end
@inline function kernel_2!(i,j,cosyear,win_NDVI,img_NDVI,img_QA,n_X,n_Y,n_B,n_Years,reference_data,d_res_3,d_res_vec_res1)
 ids=(UInt32(div(i-1,16)),UInt32(div(j-1,16)),UInt32(16),UInt32(16),UInt32(rem(i-1,16)),UInt32(rem(j-1,16)))
 Generate_NDVI_reference_body!(cosyear,win_NDVI,img_NDVI,img_QA,n_X,n_Y,n_B,n_Years,reference_data,d_res_3,d_res_vec_res1,ids...)
 return nothing
end
@inline function kernel_3!(i,j,img_NDVI,img_QA,reference_data,StartY,TotalY,Buffer_Up,Buffer_Dn,n_X,n_Y,n_B,n_Years,win,d_res)
 ids=(UInt32(div(i-1,16)),UInt32(div(j-1,16)),UInt32(16),UInt32(16),UInt32(rem(i-1,16)),UInt32(rem(j-1,16)))
 Compute_d_res_body!(img_NDVI,img_QA,reference_data,StartY,TotalY,Buffer_Up,Buffer_Dn,n_X,n_Y,n_B,n_Years,win,d_res,ids...)
 return nothing
end
@inline function kernel_4!(i,j,img_NDVI,img_QA,reference_data,StartY,TotalY,Buffer_Up,Buffer_Dn,n_X,n_Y,n_B,n_Years,win,sampcorr,snow_address,vector_out,d_vector_in,d_res,d_res_3,d_index)
 ids=(UInt32(div(i-1,16)),UInt32(div(j-1,16)),UInt32(16),UInt32(16),UInt32(rem(i-1,16)),UInt32(rem(j-1,16)))
 STSG_filter_body!(img_NDVI,img_QA,reference_data,StartY,TotalY,Buffer_Up,Buffer_Dn,n_X,n_Y,n_B,n_Years,win,sampcorr,snow_address,vector_out,d_vector_in,d_res,d_res_3,d_index,ids...)
 return nothing
end
function factory(X,Y)
 spec=JACC.launch_spec(;threads=(16,16),shmem_size=0,sync=false)
 return function(stage,args...)
  if stage==1
   JACC.parallel_for(spec,(cld(X,16)*16,cld(Y,16)*16),kernel_1!,args...)
  elseif stage==2
   JACC.parallel_for(spec,(cld(X,16)*16,cld(Y,16)*16),kernel_2!,args...)
  elseif stage==3
   JACC.parallel_for(spec,(cld(X,16)*16,cld(Y,16)*16),kernel_3!,args...)
  elseif stage==4
   JACC.parallel_for(spec,(cld(X,16)*16,cld(Y,16)*16),kernel_4!,args...)
  end
  nothing
 end
end
include("../study_common.jl")
stsg_study(factory)
