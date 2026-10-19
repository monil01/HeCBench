# Full original STSG source bodies, typed C AST lowering.
# C Float32 products/double accumulation and assignment narrowing explicit.
# Fixed local arrays diagnostic NaN/0 initialized; no native undefined-value claim.
using StaticArrays
struct Offset{T}
    array::T
    offset::Int32
end
@inline Base.getindex(a::Offset,i::Integer)=@inbounds a.array[i+a.offset]
@inline function Base.setindex!(a::Offset,x,i::Integer)
    @inbounds a.array[i+a.offset]=x
    return x
end
function Short_to_Float_body!(imgNDVI,imgQA,n_X,n_Y,n_B,n_Years,img_NDVI,img_QA,bidx::UInt32,bidy::UInt32,bdimx::UInt32,bdimy::UInt32,tidx::UInt32,tidy::UInt32)
 @inbounds begin
  local i_c1::Int32 = Int32(UInt32((UInt32((bidx*bdimx))+tidx)))
  if (i_c1>=n_X)
   return nothing
  end
  local j_c2::Int32 = Int32(UInt32((UInt32((bidy*bdimy))+tidy)))
  if (j_c2>=n_Y)
   return nothing
  end
  local k_c3::Int32 = Int32(Int32(0))
  while (k_c3<n_B)
   local y_c4::Int32 = Int32(Int32(0))
   while (y_c4<n_Years)
    local idx_c5::Int32 = Int32(Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((k_c3*n_X))*n_Y))))+Int32((Int32((Int32((y_c4*n_X))*n_Y))*n_B)))))
    img_NDVI[Int(idx_c5)+1] = Float32(Float32((Float32(imgNDVI[Int(idx_c5)+1])/Float32(10000.))))
    img_QA[Int(idx_c5)+1] = Float32(Float32(imgQA[Int(idx_c5)+1]))
    if ((((img_NDVI[Int(idx_c5)+1]<Float32((-Float32(0.2))))||(img_NDVI[Int(idx_c5)+1]>Float32(1.)))||(img_QA[Int(idx_c5)+1]<Float32((-Float32(1.)))))||(img_QA[Int(idx_c5)+1]>Float32(3.)))
     img_QA[Int(idx_c5)+1] = Float32(Float32((-Float32(1.))))
    end
    ((y_c4 += Int32(1))-Int32(1))
   end
   ((k_c3 += Int32(1))-Int32(1))
  end
 end
 nothing
end

function Generate_NDVI_reference_body!(cosyear,win_NDVI,img_NDVI,img_QA,n_X,n_Y,n_B,n_Years,reference_data,d_res_3,d_res_vec_res1,bidx::UInt32,bidy::UInt32,bdimx::UInt32,bdimy::UInt32,tidx::UInt32,tidy::UInt32)
 @inbounds begin
  local i_c1::Int32 = Int32(UInt32((UInt32((bidx*bdimx))+tidx)))
  if (i_c1>=n_X)
   return nothing
  end
  local j_c2::Int32 = Int32(UInt32((UInt32((bidy*bdimy))+tidy)))
  if (j_c2>=n_Y)
   return nothing
  end
  local res_cosyear_c3 = Offset(d_res_3,Int32(Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))*Int32((n_Years+Int32(1)))))*n_Years))))
  local y_1_c4::Int32 = Int32(Int32(0))
  while (y_1_c4<n_Years)
   local y_2_c5::Int32 = Int32(Int32((y_1_c4+Int32(1))))
   while (y_2_c5<n_Years)
    local xy_sum_c6::Float64 = Float64(Int32(0))
    local x2_sum_c7::Float64 = Float64(Int32(0))
    local y2_sum_c8::Float64 = Float64(Int32(0))
    local k_c9::Int32 = Int32(Int32(0))
    while (k_c9<n_B)
     local idx1_c10::Int32 = Int32(Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((k_c9*n_X))*n_Y))))+Int32((Int32((Int32((y_1_c4*n_X))*n_Y))*n_B)))))
     local idx2_c11::Int32 = Int32(Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((k_c9*n_X))*n_Y))))+Int32((Int32((Int32((y_2_c5*n_X))*n_Y))*n_B)))))
     if (((img_QA[Int(idx1_c10)+1]==Float32(0.))||(img_QA[Int(idx1_c10)+1]==Float32(1.)))&&((img_QA[Int(idx2_c11)+1]==Float32(0.))||(img_QA[Int(idx2_c11)+1]==Float32(1.))))
      xy_sum_c6 += Float32((img_NDVI[Int(idx1_c10)+1]*img_NDVI[Int(idx2_c11)+1]))
      x2_sum_c7 += Float32((img_NDVI[Int(idx1_c10)+1]*img_NDVI[Int(idx1_c10)+1]))
      y2_sum_c8 += Float32((img_NDVI[Int(idx2_c11)+1]*img_NDVI[Int(idx2_c11)+1]))
     end
     ((k_c9 += Int32(1))-Int32(1))
    end
    if ((x2_sum_c7!=Int32(0))&&(y2_sum_c8!=Int32(0)))
     res_cosyear_c3[Int(Int32((y_2_c5+Int32((y_1_c4*n_Years)))))+1] = Float32(Float64((xy_sum_c6/Float64(sqrt(Float64(Float64((x2_sum_c7*y2_sum_c8))))))))
     res_cosyear_c3[Int(Int32((y_1_c4+Int32((y_2_c5*n_Years)))))+1] = Float32(Float64((xy_sum_c6/Float64(sqrt(Float64(Float64((x2_sum_c7*y2_sum_c8))))))))
     res_cosyear_c3[Int(Int32((y_1_c4+Int32((y_1_c4*n_Years)))))+1] += Float64((xy_sum_c6/Float64(sqrt(Float64(Float64((x2_sum_c7*y2_sum_c8)))))))
     res_cosyear_c3[Int(Int32((y_2_c5+Int32((y_2_c5*n_Years)))))+1] += Float64((xy_sum_c6/Float64(sqrt(Float64(Float64((x2_sum_c7*y2_sum_c8)))))))
    end
    ((y_2_c5 += Int32(1))-Int32(1))
   end
   res_cosyear_c3[Int(Int32((y_1_c4+Int32((y_1_c4*n_Years)))))+1] = Float32(Float32((res_cosyear_c3[Int(Int32((y_1_c4+Int32((y_1_c4*n_Years)))))+1]/Int32((n_Years-Int32(1))))))
   res_cosyear_c3[Int(Int32((n_Years*n_Years)))+1] += res_cosyear_c3[Int(Int32((y_1_c4+Int32((y_1_c4*n_Years)))))+1]
   ((y_1_c4 += Int32(1))-Int32(1))
  end
  res_cosyear_c3[Int(Int32((n_Years*n_Years)))+1] = Float32(Float32((res_cosyear_c3[Int(Int32((n_Years*n_Years)))+1]/n_Years)))
  local y_c12::Int32 = Int32(Int32(0))
  while (y_c12<n_Years)
   res_cosyear_c3[Int(Int32((Int32((n_Years*n_Years))+Int32(1))))+1] = Float32(muladd(Float32(Float32((res_cosyear_c3[Int(Int32((y_c12+Int32((y_c12*n_Years)))))+1]-res_cosyear_c3[Int(Int32((n_Years*n_Years)))+1]))),Float32(Float32((res_cosyear_c3[Int(Int32((y_c12+Int32((y_c12*n_Years)))))+1]-res_cosyear_c3[Int(Int32((n_Years*n_Years)))+1]))),Float32(res_cosyear_c3[Int(Int32((Int32((n_Years*n_Years))+Int32(1))))+1])))
   ((y_c12 += Int32(1))-Int32(1))
  end
  res_cosyear_c3[Int(Int32((Int32((n_Years*n_Years))+Int32(1))))+1] = Float32(Float64(sqrt(Float64(Float32((res_cosyear_c3[Int(Int32((Int32((n_Years*n_Years))+Int32(1))))+1]/n_Years))))))
  res_cosyear_c3[Int(Int32((n_Years*n_Years)))+1] = Float32(Float32((res_cosyear_c3[Int(Int32((n_Years*n_Years)))+1]-res_cosyear_c3[Int(Int32((Int32((n_Years*n_Years))+Int32(1))))+1])))
  local y_1_c13::Int32 = Int32(Int32(0))
  while (y_1_c13<n_Years)
   res_cosyear_c3[Int(y_1_c13)+1] = Float32(res_cosyear_c3[Int(Int32((y_1_c13+Int32((y_1_c13*n_Years)))))+1])
   ((y_1_c13 += Int32(1))-Int32(1))
  end
  res_cosyear_c3[Int(n_Years)+1] = Float32(res_cosyear_c3[Int(Int32((n_Years*n_Years)))+1])
  local y_c14::Int32 = Int32(Int32((n_Years+Int32(1))))
  while (y_c14<Int32((Int32((n_Years+Int32(1)))*n_Years)))
   res_cosyear_c3[Int(y_c14)+1] = Float32(Int32(0))
   ((y_c14 += Int32(1))-Int32(1))
  end
  local k_c15::Int32 = Int32(Int32(0))
  while (k_c15<n_B)
   local count_img_QA_c16::Int32 = Int32(Int32(0))
   local mean_img_QA_c17::Float32 = Float32(Int32(0))
   local y_c18::Int32 = Int32(Int32(0))
   while (y_c18<n_Years)
    if ((res_cosyear_c3[Int(y_c18)+1]>=res_cosyear_c3[Int(n_Years)+1])||(res_cosyear_c3[Int(y_c18)+1]>=cosyear))
     local idx_c19::Int32 = Int32(Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((k_c15*n_X))*n_Y))))+Int32((Int32((Int32((y_c18*n_X))*n_Y))*n_B)))))
     if ((img_QA[Int(idx_c19)+1]==Float32(0.))||(img_QA[Int(idx_c19)+1]==Float32(1.)))
      ((count_img_QA_c16 += Int32(1))-Int32(1))
      mean_img_QA_c17 += img_NDVI[Int(idx_c19)+1]
     end
    end
    ((y_c18 += Int32(1))-Int32(1))
   end
   if (count_img_QA_c16>=Int32(1))
    reference_data[Int(Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((k_c15*n_X))*n_Y)))))+1] = Float32(Float32((mean_img_QA_c17/count_img_QA_c16)))
   end
   ((k_c15 += Int32(1))-Int32(1))
  end
  local n_dissimilar_year_c20::Int32 = Int32(Int32(0))
  local y_1_c21::Int32 = Int32(Int32(0))
  while (y_1_c21<n_Years)
   if ((res_cosyear_c3[Int(y_1_c21)+1]<res_cosyear_c3[Int(n_Years)+1])&&(res_cosyear_c3[Int(y_1_c21)+1]<cosyear))
    local k2_c22::Int32 = Int32(Int32((-Int32(2))))
    while (k2_c22<=Int32(2))
     if (k2_c22==Int32(0))
      ((k2_c22 += Int32(1))-Int32(1))
      continue
     end
     local xy_sum_c23::Float64 = Float64(Int32(0))
     local x2_sum_c24::Float64 = Float64(Int32(0))
     local y2_sum_c25::Float64 = Float64(Int32(0))
     local k_c26::Int32 = Int32(Int32(0))
     while (k_c26<n_B)
      if ((Int32((k_c26+k2_c22))<Int32(0))||(Int32((k_c26+k2_c22))>=n_B))
       ((k_c26 += Int32(1))-Int32(1))
       continue
      end
      local idx_c27::Int32 = Int32(Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((Int32((k_c26+k2_c22))*n_X))*n_Y))))+Int32((Int32((Int32((y_1_c21*n_X))*n_Y))*n_B)))))
      local ridx_c28::Int32 = Int32(Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((k_c26*n_X))*n_Y)))))
      if (((img_QA[Int(idx_c27)+1]==Float32(0.))||(img_QA[Int(idx_c27)+1]==Float32(1.)))&&(reference_data[Int(ridx_c28)+1]!=Float32(0.)))
       xy_sum_c23 += Float32((img_NDVI[Int(idx_c27)+1]*reference_data[Int(ridx_c28)+1]))
       x2_sum_c24 += Float32((img_NDVI[Int(idx_c27)+1]*img_NDVI[Int(idx_c27)+1]))
       y2_sum_c25 += Float32((reference_data[Int(ridx_c28)+1]*reference_data[Int(ridx_c28)+1]))
      end
      ((k_c26 += Int32(1))-Int32(1))
     end
     if ((x2_sum_c24!=Int32(0))&&(y2_sum_c25!=Int32(0)))
      res_cosyear_c3[Int(Int32((n_Years+Int32(1))))+1] = Float32(Float64((xy_sum_c23/Float64(sqrt(Float64(Float64((x2_sum_c24*y2_sum_c25))))))))
      if (res_cosyear_c3[Int(Int32((n_Years+Int32(1))))+1]>res_cosyear_c3[Int(y_1_c21)+1])
       res_cosyear_c3[Int(y_1_c21)+1] = Float32(res_cosyear_c3[Int(Int32((n_Years+Int32(1))))+1])
      end
     end
     ((k2_c22 += Int32(1))-Int32(1))
    end
    if ((res_cosyear_c3[Int(y_1_c21)+1]>=res_cosyear_c3[Int(n_Years)+1])||(res_cosyear_c3[Int(y_1_c21)+1]>=cosyear))
     ((y_1_c21 += Int32(1))-Int32(1))
     continue
    end
    local c_k_c29::Int32 = Int32(Int32(0))
    while (c_k_c29<n_B)
     local n_similar_year_c30::Int32 = Int32(Int32(0))
     local y_2_c31::Int32 = Int32(Int32(0))
     while (y_2_c31<n_Years)
      if ((res_cosyear_c3[Int(y_2_c31)+1]>=res_cosyear_c3[Int(n_Years)+1])||(((res_cosyear_c3[Int(y_2_c31)+1]>=cosyear)&&((img_QA[Int(Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((c_k_c29*n_X))*n_Y))))+Int32((Int32((Int32((y_1_c21*n_X))*n_Y))*n_B)))))+1]==Float32(0.))||(img_QA[Int(Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((c_k_c29*n_X))*n_Y))))+Int32((Int32((Int32((y_1_c21*n_X))*n_Y))*n_B)))))+1]==Float32(1.))))&&((img_QA[Int(Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((c_k_c29*n_X))*n_Y))))+Int32((Int32((Int32((y_2_c31*n_X))*n_Y))*n_B)))))+1]==Float32(0.))||(img_QA[Int(Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((c_k_c29*n_X))*n_Y))))+Int32((Int32((Int32((y_2_c31*n_X))*n_Y))*n_B)))))+1]==Float32(1.)))))
       local xy_sum_c32::Float64 = Float64(Int32(0))
       local x2_sum_c33::Float64 = Float64(Int32(0))
       local y2_sum_c34::Float64 = Float64(Int32(0))
       local k_c35::Int32 = Int32(Int32(1))
       local n_c36::Int32 = Int32(Int32(0))
       while ((n_c36<Int32((Int32((win_NDVI*Int32(2)))+Int32(1))))&&((Int32((c_k_c29+k_c35))<n_B)||(Int32((c_k_c29-k_c35))>=Int32(0))))
        if (((Int32((c_k_c29-k_c35))>=Int32(0))&&((img_QA[Int(Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((Int32((c_k_c29-k_c35))*n_X))*n_Y))))+Int32((Int32((Int32((y_1_c21*n_X))*n_Y))*n_B)))))+1]==Float32(0.))||(img_QA[Int(Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((Int32((c_k_c29-k_c35))*n_X))*n_Y))))+Int32((Int32((Int32((y_1_c21*n_X))*n_Y))*n_B)))))+1]==Float32(1.))))&&((img_QA[Int(Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((Int32((c_k_c29-k_c35))*n_X))*n_Y))))+Int32((Int32((Int32((y_2_c31*n_X))*n_Y))*n_B)))))+1]==Float32(0.))||(img_QA[Int(Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((Int32((c_k_c29-k_c35))*n_X))*n_Y))))+Int32((Int32((Int32((y_2_c31*n_X))*n_Y))*n_B)))))+1]==Float32(1.))))
         xy_sum_c32 += Float32((img_NDVI[Int(Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((Int32((c_k_c29-k_c35))*n_X))*n_Y))))+Int32((Int32((Int32((y_1_c21*n_X))*n_Y))*n_B)))))+1]*img_NDVI[Int(Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((Int32((c_k_c29-k_c35))*n_X))*n_Y))))+Int32((Int32((Int32((y_2_c31*n_X))*n_Y))*n_B)))))+1]))
         x2_sum_c33 += Float32((img_NDVI[Int(Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((Int32((c_k_c29-k_c35))*n_X))*n_Y))))+Int32((Int32((Int32((y_1_c21*n_X))*n_Y))*n_B)))))+1]*img_NDVI[Int(Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((Int32((c_k_c29-k_c35))*n_X))*n_Y))))+Int32((Int32((Int32((y_1_c21*n_X))*n_Y))*n_B)))))+1]))
         y2_sum_c34 += Float32((img_NDVI[Int(Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((Int32((c_k_c29-k_c35))*n_X))*n_Y))))+Int32((Int32((Int32((y_2_c31*n_X))*n_Y))*n_B)))))+1]*img_NDVI[Int(Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((Int32((c_k_c29-k_c35))*n_X))*n_Y))))+Int32((Int32((Int32((y_2_c31*n_X))*n_Y))*n_B)))))+1]))
         ((n_c36 += Int32(1))-Int32(1))
        end
        if (((Int32((c_k_c29+k_c35))<n_B)&&((img_QA[Int(Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((Int32((c_k_c29+k_c35))*n_X))*n_Y))))+Int32((Int32((Int32((y_1_c21*n_X))*n_Y))*n_B)))))+1]==Float32(0.))||(img_QA[Int(Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((Int32((c_k_c29+k_c35))*n_X))*n_Y))))+Int32((Int32((Int32((y_1_c21*n_X))*n_Y))*n_B)))))+1]==Float32(1.))))&&((img_QA[Int(Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((Int32((c_k_c29+k_c35))*n_X))*n_Y))))+Int32((Int32((Int32((y_2_c31*n_X))*n_Y))*n_B)))))+1]==Float32(0.))||(img_QA[Int(Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((Int32((c_k_c29+k_c35))*n_X))*n_Y))))+Int32((Int32((Int32((y_2_c31*n_X))*n_Y))*n_B)))))+1]==Float32(1.))))
         xy_sum_c32 += Float32((img_NDVI[Int(Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((Int32((c_k_c29+k_c35))*n_X))*n_Y))))+Int32((Int32((Int32((y_1_c21*n_X))*n_Y))*n_B)))))+1]*img_NDVI[Int(Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((Int32((c_k_c29+k_c35))*n_X))*n_Y))))+Int32((Int32((Int32((y_2_c31*n_X))*n_Y))*n_B)))))+1]))
         x2_sum_c33 += Float32((img_NDVI[Int(Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((Int32((c_k_c29+k_c35))*n_X))*n_Y))))+Int32((Int32((Int32((y_1_c21*n_X))*n_Y))*n_B)))))+1]*img_NDVI[Int(Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((Int32((c_k_c29+k_c35))*n_X))*n_Y))))+Int32((Int32((Int32((y_1_c21*n_X))*n_Y))*n_B)))))+1]))
         y2_sum_c34 += Float32((img_NDVI[Int(Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((Int32((c_k_c29+k_c35))*n_X))*n_Y))))+Int32((Int32((Int32((y_2_c31*n_X))*n_Y))*n_B)))))+1]*img_NDVI[Int(Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((Int32((c_k_c29+k_c35))*n_X))*n_Y))))+Int32((Int32((Int32((y_2_c31*n_X))*n_Y))*n_B)))))+1]))
         ((n_c36 += Int32(1))-Int32(1))
        end
        ((k_c35 += Int32(1))-Int32(1))
       end
       if ((x2_sum_c33!=Int32(0))&&(y2_sum_c34!=Int32(0)))
        res_cosyear_c3[Int(Int32((Int32((c_k_c29+Int32((n_dissimilar_year_c20*n_B))))+n_Years)))+1] += Float64((xy_sum_c32/Float64(sqrt(Float64(Float64((x2_sum_c33*y2_sum_c34)))))))
        ((n_similar_year_c30 += Int32(1))-Int32(1))
       end
      end
      ((y_2_c31 += Int32(1))-Int32(1))
     end
     if (n_similar_year_c30!=Int32(0))
      res_cosyear_c3[Int(Int32((Int32((Int32((c_k_c29+Int32((n_dissimilar_year_c20*n_B))))+n_Years))+Int32(2))))+1] = Float32(Float32((res_cosyear_c3[Int(Int32((Int32((Int32((c_k_c29+Int32((n_dissimilar_year_c20*n_B))))+n_Years))+Int32(2))))+1]/n_similar_year_c30)))
     end
     ((c_k_c29 += Int32(1))-Int32(1))
    end
    ((n_dissimilar_year_c20 += Int32(1))-Int32(1))
   end
   ((y_1_c21 += Int32(1))-Int32(1))
  end
  local count_vec_res1_c37::Int32 = Int32(Int32(0))
  local res_vec_res1_c38 = Offset(d_res_vec_res1,Int32(Int32((Int32((i_c1+Int32((j_c2*n_X))))*n_B))))
  local k_c39::Int32 = Int32(Int32(0))
  while (k_c39<n_B)
   local count_img_QA_c40::Int32 = Int32(Int32(0))
   local mean_img_QA_c41::Float32 = Float32(Int32(0))
   n_dissimilar_year_c20 = Int32(Int32(0))
   local y_c42::Int32 = Int32(Int32(0))
   while (y_c42<n_Years)
    if ((res_cosyear_c3[Int(y_c42)+1]>=res_cosyear_c3[Int(n_Years)+1])||(res_cosyear_c3[Int(y_c42)+1]>=cosyear))
     if ((img_QA[Int(Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((k_c39*n_X))*n_Y))))+Int32((Int32((Int32((y_c42*n_X))*n_Y))*n_B)))))+1]==Float32(0.))||(img_QA[Int(Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((k_c39*n_X))*n_Y))))+Int32((Int32((Int32((y_c42*n_X))*n_Y))*n_B)))))+1]==Float32(1.)))
      ((count_img_QA_c40 += Int32(1))-Int32(1))
      mean_img_QA_c41 += img_NDVI[Int(Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((k_c39*n_X))*n_Y))))+Int32((Int32((Int32((y_c42*n_X))*n_Y))*n_B)))))+1]
     end
    else
     if (((res_cosyear_c3[Int(Int32((Int32((Int32((k_c39+Int32((n_dissimilar_year_c20*n_B))))+n_Years))+Int32(2))))+1]>=res_cosyear_c3[Int(n_Years)+1])||(res_cosyear_c3[Int(Int32((Int32((Int32((k_c39+Int32((n_dissimilar_year_c20*n_B))))+n_Years))+Int32(2))))+1]>=cosyear))&&((img_QA[Int(Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((k_c39*n_X))*n_Y))))+Int32((Int32((Int32((y_c42*n_X))*n_Y))*n_B)))))+1]==Float32(0.))||(img_QA[Int(Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((k_c39*n_X))*n_Y))))+Int32((Int32((Int32((y_c42*n_X))*n_Y))*n_B)))))+1]==Float32(1.))))
      ((count_img_QA_c40 += Int32(1))-Int32(1))
      mean_img_QA_c41 += img_NDVI[Int(Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((k_c39*n_X))*n_Y))))+Int32((Int32((Int32((y_c42*n_X))*n_Y))*n_B)))))+1]
     end
     ((n_dissimilar_year_c20 += Int32(1))-Int32(1))
    end
    ((y_c42 += Int32(1))-Int32(1))
   end
   if (count_img_QA_c40>=Int32(1))
    reference_data[Int(Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((k_c39*n_X))*n_Y)))))+1] = Float32(Float32((mean_img_QA_c41/count_img_QA_c40)))
    res_vec_res1_c38[Int(((count_vec_res1_c37 += Int32(1))-Int32(1)))+1] = Int32(k_c39)
   end
   ((k_c39 += Int32(1))-Int32(1))
  end
  if ((count_vec_res1_c37<n_B)&&(count_vec_res1_c37>Int32(1)))
   local k_c43::Float64 = Float64(NaN)
   local x_c44::Int32 = Int32(Int32(0))
   local l_c45::Int32 = Int32(Int32(0))
   if (res_vec_res1_c38[Int(Int32((count_vec_res1_c37-Int32(1))))+1]!=Int32((n_B-Int32(1))))
    k_c43 = Float64(Float32((Float32((reference_data[Int(Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((res_vec_res1_c38[Int(Int32((count_vec_res1_c37-Int32(1))))+1]*n_X))*n_Y)))))+1]-reference_data[Int(Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((res_vec_res1_c38[Int(Int32((count_vec_res1_c37-Int32(2))))+1]*n_X))*n_Y)))))+1]))/Int32((res_vec_res1_c38[Int(Int32((count_vec_res1_c37-Int32(1))))+1]-res_vec_res1_c38[Int(Int32((count_vec_res1_c37-Int32(2))))+1])))))
    reference_data[Int(Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((Int32((n_B-Int32(1)))*n_X))*n_Y)))))+1] = Float32(muladd(Float64(k_c43),Float64(Int32((Int32((n_B-Int32(1)))-res_vec_res1_c38[Int(Int32((count_vec_res1_c37-Int32(1))))+1]))),Float64(reference_data[Int(Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((res_vec_res1_c38[Int(Int32((count_vec_res1_c37-Int32(1))))+1]*n_X))*n_Y)))))+1])))
    res_vec_res1_c38[Int(((count_vec_res1_c37 += Int32(1))-Int32(1)))+1] = Int32(Int32((n_B-Int32(1))))
   end
   local count_res_vec_res1_c46::Int32 = Int32(count_vec_res1_c37)
   while ((count_vec_res1_c37<n_B)&&(x_c44<Int32((count_res_vec_res1_c46-Int32(1)))))
    l_c45 = Int32(Int32((res_vec_res1_c38[Int(Int32((x_c44+Int32(1))))+1]-res_vec_res1_c38[Int(x_c44)+1])))
    local n_c47::Int32 = Int32(Int32(1))
    while (l_c45>Int32(1))
     k_c43 = Float64(Float32((Float32((reference_data[Int(Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((res_vec_res1_c38[Int(Int32((x_c44+Int32(1))))+1]*n_X))*n_Y)))))+1]-reference_data[Int(Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((res_vec_res1_c38[Int(x_c44)+1]*n_X))*n_Y)))))+1]))/Int32((res_vec_res1_c38[Int(Int32((x_c44+Int32(1))))+1]-res_vec_res1_c38[Int(x_c44)+1])))))
     reference_data[Int(Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((Int32((res_vec_res1_c38[Int(x_c44)+1]+n_c47))*n_X))*n_Y)))))+1] = Float32(muladd(Float64(k_c43),Float64(n_c47),Float64(reference_data[Int(Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((res_vec_res1_c38[Int(x_c44)+1]*n_X))*n_Y)))))+1])))
     ((count_vec_res1_c37 += Int32(1))-Int32(1))
     ((n_c47 += Int32(1))-Int32(1))
     ((l_c45 -= Int32(1))+Int32(1))
    end
    ((x_c44 += Int32(1))-Int32(1))
   end
   if (res_vec_res1_c38[Int(Int32(0))+1]!=Int32(0))
    k_c43 = Float64(Float32((Float32((reference_data[Int(Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((res_vec_res1_c38[Int(Int32(1))+1]*n_X))*n_Y)))))+1]-reference_data[Int(Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((res_vec_res1_c38[Int(Int32(0))+1]*n_X))*n_Y)))))+1]))/Int32((res_vec_res1_c38[Int(Int32(1))+1]-res_vec_res1_c38[Int(Int32(0))+1])))))
    l_c45 = Int32(res_vec_res1_c38[Int(Int32(0))+1])
    local n_c48::Int32 = Int32(Int32(0))
    while true
     reference_data[Int(Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((n_c48*n_X))*n_Y)))))+1] = Float32(muladd((-Float64(k_c43)),Float64(l_c45),Float64(reference_data[Int(Int32((Int32((i_c1+Int32((j_c2*n_X))))+Int32((Int32((res_vec_res1_c38[Int(Int32(0))+1]*n_X))*n_Y)))))+1])))
     ((n_c48 += Int32(1))-Int32(1))
     ((l_c45 -= Int32(1))+Int32(1))
     if !((l_c45>=Int32(1)));break;end
    end
   end
  end
 end
 nothing
end

function Compute_d_res_body!(img_NDVI,img_QA,reference_data,StartY,TotalY,Buffer_Up,Buffer_Dn,n_X,n_Y,n_B,n_Years,win,d_res,bidx::UInt32,bidy::UInt32,bdimx::UInt32,bdimy::UInt32,tidx::UInt32,tidy::UInt32)
 @inbounds begin
  local i_c1::Int32 = Int32(UInt32((UInt32((bidx*bdimx))+tidx)))
  if (i_c1>=n_X)
   return nothing
  end
  local j_c2::Int32 = Int32(UInt32((UInt32((bidy*bdimy))+tidy)))
  if (j_c2>=Int32((Int32((n_Y-Buffer_Dn))-Buffer_Up)))
   return nothing
  end
  local corr_res_c3 = Offset(d_res,Int32(Int32((Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(4)))))
  local Slope_res_c4 = Offset(d_res,Int32(Int32((Int32((Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(4)))+Int32((Int32((Int32((Int32(2)*win))+Int32(1)))*Int32((Int32((Int32(2)*win))+Int32(1)))))))))
  local Intercept_res_c5 = Offset(d_res,Int32(Int32((Int32((Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(4)))+Int32((Int32((Int32((Int32((Int32(2)*win))+Int32(1)))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(2)))))))
  local new_corr_similar_res_c6 = Offset(d_res,Int32(Int32((Int32((Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(4)))+Int32((Int32((Int32((Int32((Int32(2)*win))+Int32(1)))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(3)))))))
  local sj_c7::Int32 = Int32(Int32(0))
  while (sj_c7<=win)
   local si_c8::Int32 = Int32(Int32((Int32((-Int32(1)))*win)))
   while (si_c8<=win)
    if ((((Int32((i_c1+si_c8))>=Int32(0))&&(Int32((i_c1+si_c8))<n_X))&&(Int32((Int32((j_c2+sj_c7))+StartY))>=Int32(0)))&&(Int32((Int32((j_c2+sj_c7))+StartY))<TotalY))
     local x_sum_c9::Float64 = Float64(Int32(0))
     local y_sum_c10::Float64 = Float64(Int32(0))
     local x_mean_c11::Float64 = Float64(Int32(0))
     local y_mean_c12::Float64 = Float64(Int32(0))
     local xy_sum_c13::Float64 = Float64(Int32(0))
     local x2_sum_c14::Float64 = Float64(Int32(0))
     local y2_sum_c15::Float64 = Float64(Int32(0))
     local k_c16::Int32 = Int32(Int32(0))
     while (k_c16<n_B)
      x_sum_c9 += reference_data[Int(Int32((Int32((Int32((i_c1+si_c8))+Int32((Int32((Int32((j_c2+sj_c7))+Buffer_Up))*n_X))))+Int32((Int32((k_c16*n_X))*n_Y)))))+1]
      y_sum_c10 += reference_data[Int(Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c16*n_X))*n_Y)))))+1]
      ((k_c16 += Int32(1))-Int32(1))
     end
     x_mean_c11 = Float64(Float64((x_sum_c9/n_B)))
     y_mean_c12 = Float64(Float64((y_sum_c10/n_B)))
     local k_c17::Int32 = Int32(Int32(0))
     while (k_c17<n_B)
      xy_sum_c13 = Float64(muladd(Float64(Float64((reference_data[Int(Int32((Int32((Int32((i_c1+si_c8))+Int32((Int32((Int32((j_c2+sj_c7))+Buffer_Up))*n_X))))+Int32((Int32((k_c17*n_X))*n_Y)))))+1]-x_mean_c11))),Float64(Float64((reference_data[Int(Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c17*n_X))*n_Y)))))+1]-y_mean_c12))),Float64(xy_sum_c13)))
      x2_sum_c14 = Float64(muladd(Float64(Float64((reference_data[Int(Int32((Int32((Int32((i_c1+si_c8))+Int32((Int32((Int32((j_c2+sj_c7))+Buffer_Up))*n_X))))+Int32((Int32((k_c17*n_X))*n_Y)))))+1]-x_mean_c11))),Float64(Float64((reference_data[Int(Int32((Int32((Int32((i_c1+si_c8))+Int32((Int32((Int32((j_c2+sj_c7))+Buffer_Up))*n_X))))+Int32((Int32((k_c17*n_X))*n_Y)))))+1]-x_mean_c11))),Float64(x2_sum_c14)))
      y2_sum_c15 = Float64(muladd(Float64(Float64((reference_data[Int(Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c17*n_X))*n_Y)))))+1]-y_mean_c12))),Float64(Float64((reference_data[Int(Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c17*n_X))*n_Y)))))+1]-y_mean_c12))),Float64(y2_sum_c15)))
      ((k_c17 += Int32(1))-Int32(1))
     end
     corr_res_c3[Int(Int32((Int32((si_c8+win))+Int32((Int32((sj_c7+win))*Int32((Int32((Int32(2)*win))+Int32(1))))))))+1] = Float32(Float64((xy_sum_c13/Float64(sqrt(Float64(Float64((x2_sum_c14*y2_sum_c15))))))))
     d_res[Int(Int32((Int32((Int32((Int32((Int32((Int32((Int32((Int32((i_c1+si_c8))+Int32((Int32((j_c2+sj_c7))*n_X))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(4)))-si_c8))+win))+Int32((Int32((Int32((Int32((-Int32(1)))*sj_c7))+win))*Int32((Int32((Int32(2)*win))+Int32(1))))))))+1] = Float32(Float64((xy_sum_c13/Float64(sqrt(Float64(Float64((x2_sum_c14*y2_sum_c15))))))))
     local count_tempQA_c18::Int32 = Int32(Int32(0))
     x_sum_c9 = Float64(Int32(0))
     y_sum_c10 = Float64(Int32(0))
     x_mean_c11 = Float64(Int32(0))
     y_mean_c12 = Float64(Int32(0))
     xy_sum_c13 = Float64(Int32(0))
     x2_sum_c14 = Float64(Int32(0))
     y2_sum_c15 = Float64(Int32(0))
     local k_c19::Int32 = Int32(Int32(0))
     while (k_c19<n_B)
      local y_c20::Int32 = Int32(Int32(0))
      while (y_c20<n_Years)
       if (((((img_QA[Int(Int32((Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c19*n_X))*n_Y))))+Int32((Int32((Int32((y_c20*n_X))*n_Y))*n_B)))))+1]==Float32(0.))||(img_QA[Int(Int32((Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c19*n_X))*n_Y))))+Int32((Int32((Int32((y_c20*n_X))*n_Y))*n_B)))))+1]==Float32(1.)))&&((img_QA[Int(Int32((Int32((Int32((Int32((i_c1+si_c8))+Int32((Int32((Int32((j_c2+sj_c7))+Buffer_Up))*n_X))))+Int32((Int32((k_c19*n_X))*n_Y))))+Int32((Int32((Int32((y_c20*n_X))*n_Y))*n_B)))))+1]==Float32(0.))||(img_QA[Int(Int32((Int32((Int32((Int32((i_c1+si_c8))+Int32((Int32((Int32((j_c2+sj_c7))+Buffer_Up))*n_X))))+Int32((Int32((k_c19*n_X))*n_Y))))+Int32((Int32((Int32((y_c20*n_X))*n_Y))*n_B)))))+1]==Float32(1.))))&&(reference_data[Int(Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c19*n_X))*n_Y)))))+1]!=Float32(0.)))&&(reference_data[Int(Int32((Int32((Int32((i_c1+si_c8))+Int32((Int32((Int32((j_c2+sj_c7))+Buffer_Up))*n_X))))+Int32((Int32((k_c19*n_X))*n_Y)))))+1]!=Float32(0.)))
        ((count_tempQA_c18 += Int32(1))-Int32(1))
        x_sum_c9 += Float32((img_NDVI[Int(Int32((Int32((Int32((Int32((i_c1+si_c8))+Int32((Int32((Int32((j_c2+sj_c7))+Buffer_Up))*n_X))))+Int32((Int32((k_c19*n_X))*n_Y))))+Int32((Int32((Int32((y_c20*n_X))*n_Y))*n_B)))))+1]/reference_data[Int(Int32((Int32((Int32((i_c1+si_c8))+Int32((Int32((Int32((j_c2+sj_c7))+Buffer_Up))*n_X))))+Int32((Int32((k_c19*n_X))*n_Y)))))+1]))
        y_sum_c10 += Float32((img_NDVI[Int(Int32((Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c19*n_X))*n_Y))))+Int32((Int32((Int32((y_c20*n_X))*n_Y))*n_B)))))+1]/reference_data[Int(Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c19*n_X))*n_Y)))))+1]))
        xy_sum_c13 = Float64(muladd(Float64(Float64((Float32((img_NDVI[Int(Int32((Int32((Int32((Int32((i_c1+si_c8))+Int32((Int32((Int32((j_c2+sj_c7))+Buffer_Up))*n_X))))+Int32((Int32((k_c19*n_X))*n_Y))))+Int32((Int32((Int32((y_c20*n_X))*n_Y))*n_B)))))+1]/reference_data[Int(Int32((Int32((Int32((i_c1+si_c8))+Int32((Int32((Int32((j_c2+sj_c7))+Buffer_Up))*n_X))))+Int32((Int32((k_c19*n_X))*n_Y)))))+1]))-x_mean_c11))),Float64(Float64((Float32((img_NDVI[Int(Int32((Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c19*n_X))*n_Y))))+Int32((Int32((Int32((y_c20*n_X))*n_Y))*n_B)))))+1]/reference_data[Int(Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c19*n_X))*n_Y)))))+1]))-y_mean_c12))),Float64(xy_sum_c13)))
        x2_sum_c14 = Float64(muladd(Float64(Float64((Float32((img_NDVI[Int(Int32((Int32((Int32((Int32((i_c1+si_c8))+Int32((Int32((Int32((j_c2+sj_c7))+Buffer_Up))*n_X))))+Int32((Int32((k_c19*n_X))*n_Y))))+Int32((Int32((Int32((y_c20*n_X))*n_Y))*n_B)))))+1]/reference_data[Int(Int32((Int32((Int32((i_c1+si_c8))+Int32((Int32((Int32((j_c2+sj_c7))+Buffer_Up))*n_X))))+Int32((Int32((k_c19*n_X))*n_Y)))))+1]))-x_mean_c11))),Float64(Float64((Float32((img_NDVI[Int(Int32((Int32((Int32((Int32((i_c1+si_c8))+Int32((Int32((Int32((j_c2+sj_c7))+Buffer_Up))*n_X))))+Int32((Int32((k_c19*n_X))*n_Y))))+Int32((Int32((Int32((y_c20*n_X))*n_Y))*n_B)))))+1]/reference_data[Int(Int32((Int32((Int32((i_c1+si_c8))+Int32((Int32((Int32((j_c2+sj_c7))+Buffer_Up))*n_X))))+Int32((Int32((k_c19*n_X))*n_Y)))))+1]))-x_mean_c11))),Float64(x2_sum_c14)))
        y2_sum_c15 = Float64(muladd(Float64(Float64((Float32((img_NDVI[Int(Int32((Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c19*n_X))*n_Y))))+Int32((Int32((Int32((y_c20*n_X))*n_Y))*n_B)))))+1]/reference_data[Int(Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c19*n_X))*n_Y)))))+1]))-y_mean_c12))),Float64(Float64((Float32((img_NDVI[Int(Int32((Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c19*n_X))*n_Y))))+Int32((Int32((Int32((y_c20*n_X))*n_Y))*n_B)))))+1]/reference_data[Int(Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c19*n_X))*n_Y)))))+1]))-y_mean_c12))),Float64(y2_sum_c15)))
       end
       ((y_c20 += Int32(1))-Int32(1))
      end
      ((k_c19 += Int32(1))-Int32(1))
     end
     if (count_tempQA_c18>=Int32(30))
      Slope_res_c4[Int(Int32((Int32((si_c8+win))+Int32((Int32((sj_c7+win))*Int32((Int32((Int32(2)*win))+Int32(1))))))))+1] = Float32(Float64((muladd(Float64(xy_sum_c13),Float64(count_tempQA_c18),(-Float64(Float64((x_sum_c9*y_sum_c10)))))/muladd(Float64(x2_sum_c14),Float64(count_tempQA_c18),(-Float64(Float64((x_sum_c9*x_sum_c9))))))))
      Intercept_res_c5[Int(Int32((Int32((si_c8+win))+Int32((Int32((sj_c7+win))*Int32((Int32((Int32(2)*win))+Int32(1))))))))+1] = Float32(Float64((muladd(Float64(x2_sum_c14),Float64(y_sum_c10),(-Float64(Float64((x_sum_c9*xy_sum_c13)))))/muladd(Float64(x2_sum_c14),Float64(count_tempQA_c18),(-Float64(Float64((x_sum_c9*x_sum_c9))))))))
      d_res[Int(Int32((Int32((Int32((Int32((Int32((Int32((Int32((Int32((Int32((i_c1+si_c8))+Int32((Int32((j_c2+sj_c7))*n_X))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(4)))+Int32((Int32((Int32((Int32(2)*win))+Int32(1)))*Int32((Int32((Int32(2)*win))+Int32(1)))))))-si_c8))+win))+Int32((Int32((Int32((Int32((-Int32(1)))*sj_c7))+win))*Int32((Int32((Int32(2)*win))+Int32(1))))))))+1] = Float32(Float64((muladd(Float64(xy_sum_c13),Float64(count_tempQA_c18),(-Float64(Float64((y_sum_c10*x_sum_c9)))))/muladd(Float64(y2_sum_c15),Float64(count_tempQA_c18),(-Float64(Float64((y_sum_c10*y_sum_c10))))))))
      d_res[Int(Int32((Int32((Int32((Int32((Int32((Int32((Int32((Int32((Int32((i_c1+si_c8))+Int32((Int32((j_c2+sj_c7))*n_X))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(4)))+Int32((Int32((Int32((Int32((Int32(2)*win))+Int32(1)))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(2)))))-si_c8))+win))+Int32((Int32((Int32((Int32((-Int32(1)))*sj_c7))+win))*Int32((Int32((Int32(2)*win))+Int32(1))))))))+1] = Float32(Float64((muladd(Float64(y2_sum_c15),Float64(x_sum_c9),(-Float64(Float64((y_sum_c10*xy_sum_c13)))))/muladd(Float64(y2_sum_c15),Float64(count_tempQA_c18),(-Float64(Float64((y_sum_c10*y_sum_c10))))))))
      x_mean_c11 = Float64(Float64((x_sum_c9/count_tempQA_c18)))
      y_mean_c12 = Float64(Float64((y_sum_c10/count_tempQA_c18)))
      xy_sum_c13 = Float64(Int32(0))
      x2_sum_c14 = Float64(Int32(0))
      y2_sum_c15 = Float64(Int32(0))
      local k_c21::Int32 = Int32(Int32(0))
      while (k_c21<n_B)
       local y_c22::Int32 = Int32(Int32(0))
       while (y_c22<n_Years)
        if (((((img_QA[Int(Int32((Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c21*n_X))*n_Y))))+Int32((Int32((Int32((y_c22*n_X))*n_Y))*n_B)))))+1]==Float32(0.))||(img_QA[Int(Int32((Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c21*n_X))*n_Y))))+Int32((Int32((Int32((y_c22*n_X))*n_Y))*n_B)))))+1]==Float32(1.)))&&((img_QA[Int(Int32((Int32((Int32((Int32((i_c1+si_c8))+Int32((Int32((Int32((j_c2+sj_c7))+Buffer_Up))*n_X))))+Int32((Int32((k_c21*n_X))*n_Y))))+Int32((Int32((Int32((y_c22*n_X))*n_Y))*n_B)))))+1]==Float32(0.))||(img_QA[Int(Int32((Int32((Int32((Int32((i_c1+si_c8))+Int32((Int32((Int32((j_c2+sj_c7))+Buffer_Up))*n_X))))+Int32((Int32((k_c21*n_X))*n_Y))))+Int32((Int32((Int32((y_c22*n_X))*n_Y))*n_B)))))+1]==Float32(1.))))&&(reference_data[Int(Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c21*n_X))*n_Y)))))+1]!=Float32(0.)))&&(reference_data[Int(Int32((Int32((Int32((i_c1+si_c8))+Int32((Int32((Int32((j_c2+sj_c7))+Buffer_Up))*n_X))))+Int32((Int32((k_c21*n_X))*n_Y)))))+1]!=Float32(0.)))
         xy_sum_c13 = Float64(muladd(Float64(Float64((Float32((img_NDVI[Int(Int32((Int32((Int32((Int32((i_c1+si_c8))+Int32((Int32((Int32((j_c2+sj_c7))+Buffer_Up))*n_X))))+Int32((Int32((k_c21*n_X))*n_Y))))+Int32((Int32((Int32((y_c22*n_X))*n_Y))*n_B)))))+1]/reference_data[Int(Int32((Int32((Int32((i_c1+si_c8))+Int32((Int32((Int32((j_c2+sj_c7))+Buffer_Up))*n_X))))+Int32((Int32((k_c21*n_X))*n_Y)))))+1]))-x_mean_c11))),Float64(Float64((Float32((img_NDVI[Int(Int32((Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c21*n_X))*n_Y))))+Int32((Int32((Int32((y_c22*n_X))*n_Y))*n_B)))))+1]/reference_data[Int(Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c21*n_X))*n_Y)))))+1]))-y_mean_c12))),Float64(xy_sum_c13)))
         x2_sum_c14 = Float64(muladd(Float64(Float64((Float32((img_NDVI[Int(Int32((Int32((Int32((Int32((i_c1+si_c8))+Int32((Int32((Int32((j_c2+sj_c7))+Buffer_Up))*n_X))))+Int32((Int32((k_c21*n_X))*n_Y))))+Int32((Int32((Int32((y_c22*n_X))*n_Y))*n_B)))))+1]/reference_data[Int(Int32((Int32((Int32((i_c1+si_c8))+Int32((Int32((Int32((j_c2+sj_c7))+Buffer_Up))*n_X))))+Int32((Int32((k_c21*n_X))*n_Y)))))+1]))-x_mean_c11))),Float64(Float64((Float32((img_NDVI[Int(Int32((Int32((Int32((Int32((i_c1+si_c8))+Int32((Int32((Int32((j_c2+sj_c7))+Buffer_Up))*n_X))))+Int32((Int32((k_c21*n_X))*n_Y))))+Int32((Int32((Int32((y_c22*n_X))*n_Y))*n_B)))))+1]/reference_data[Int(Int32((Int32((Int32((i_c1+si_c8))+Int32((Int32((Int32((j_c2+sj_c7))+Buffer_Up))*n_X))))+Int32((Int32((k_c21*n_X))*n_Y)))))+1]))-x_mean_c11))),Float64(x2_sum_c14)))
         y2_sum_c15 = Float64(muladd(Float64(Float64((Float32((img_NDVI[Int(Int32((Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c21*n_X))*n_Y))))+Int32((Int32((Int32((y_c22*n_X))*n_Y))*n_B)))))+1]/reference_data[Int(Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c21*n_X))*n_Y)))))+1]))-y_mean_c12))),Float64(Float64((Float32((img_NDVI[Int(Int32((Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c21*n_X))*n_Y))))+Int32((Int32((Int32((y_c22*n_X))*n_Y))*n_B)))))+1]/reference_data[Int(Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c21*n_X))*n_Y)))))+1]))-y_mean_c12))),Float64(y2_sum_c15)))
        end
        ((y_c22 += Int32(1))-Int32(1))
       end
       ((k_c21 += Int32(1))-Int32(1))
      end
      new_corr_similar_res_c6[Int(Int32((Int32((si_c8+win))+Int32((Int32((sj_c7+win))*Int32((Int32((Int32(2)*win))+Int32(1))))))))+1] = Float32(Float64((xy_sum_c13/Float64(sqrt(Float64(Float64((x2_sum_c14*y2_sum_c15))))))))
      d_res[Int(Int32((Int32((Int32((Int32((Int32((Int32((Int32((Int32((Int32((i_c1+si_c8))+Int32((Int32((j_c2+sj_c7))*n_X))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(4)))+Int32((Int32((Int32((Int32((Int32(2)*win))+Int32(1)))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(3)))))-si_c8))+win))+Int32((Int32((Int32((Int32((-Int32(1)))*sj_c7))+win))*Int32((Int32((Int32(2)*win))+Int32(1))))))))+1] = Float32(Float64((xy_sum_c13/Float64(sqrt(Float64(Float64((x2_sum_c14*y2_sum_c15))))))))
     end
     if (reference_data[Int(Int32((Int32((Int32((i_c1+si_c8))+Int32((Int32((Int32((j_c2+sj_c7))+Buffer_Up))*n_X))))+Int32((Int32((Int32(3)*n_X))*n_Y)))))+1]==Int32(0))
      corr_res_c3[Int(Int32((Int32((si_c8+win))+Int32((Int32((sj_c7+win))*Int32((Int32((Int32(2)*win))+Int32(1))))))))+1] = Float32(Int32(0))
      Slope_res_c4[Int(Int32((Int32((si_c8+win))+Int32((Int32((sj_c7+win))*Int32((Int32((Int32(2)*win))+Int32(1))))))))+1] = Float32(Int32(0))
      Intercept_res_c5[Int(Int32((Int32((si_c8+win))+Int32((Int32((sj_c7+win))*Int32((Int32((Int32(2)*win))+Int32(1))))))))+1] = Float32(Int32(0))
      new_corr_similar_res_c6[Int(Int32((Int32((si_c8+win))+Int32((Int32((sj_c7+win))*Int32((Int32((Int32(2)*win))+Int32(1))))))))+1] = Float32(Int32(0))
     end
     if (reference_data[Int(Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((Int32(3)*n_X))*n_Y)))))+1]==Int32(0))
      d_res[Int(Int32((Int32((Int32((Int32((Int32((Int32((Int32((Int32((i_c1+si_c8))+Int32((Int32((j_c2+sj_c7))*n_X))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(4)))-si_c8))+win))+Int32((Int32((Int32((Int32((-Int32(1)))*sj_c7))+win))*Int32((Int32((Int32(2)*win))+Int32(1))))))))+1] = Float32(Int32(0))
      d_res[Int(Int32((Int32((Int32((Int32((Int32((Int32((Int32((Int32((Int32((i_c1+si_c8))+Int32((Int32((j_c2+sj_c7))*n_X))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(4)))+Int32((Int32((Int32((Int32(2)*win))+Int32(1)))*Int32((Int32((Int32(2)*win))+Int32(1)))))))-si_c8))+win))+Int32((Int32((Int32((Int32((-Int32(1)))*sj_c7))+win))*Int32((Int32((Int32(2)*win))+Int32(1))))))))+1] = Float32(Int32(0))
      d_res[Int(Int32((Int32((Int32((Int32((Int32((Int32((Int32((Int32((Int32((i_c1+si_c8))+Int32((Int32((j_c2+sj_c7))*n_X))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(4)))+Int32((Int32((Int32((Int32((Int32(2)*win))+Int32(1)))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(2)))))-si_c8))+win))+Int32((Int32((Int32((Int32((-Int32(1)))*sj_c7))+win))*Int32((Int32((Int32(2)*win))+Int32(1))))))))+1] = Float32(Int32(0))
      d_res[Int(Int32((Int32((Int32((Int32((Int32((Int32((Int32((Int32((Int32((i_c1+si_c8))+Int32((Int32((j_c2+sj_c7))*n_X))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(4)))+Int32((Int32((Int32((Int32((Int32(2)*win))+Int32(1)))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(3)))))-si_c8))+win))+Int32((Int32((Int32((Int32((-Int32(1)))*sj_c7))+win))*Int32((Int32((Int32(2)*win))+Int32(1))))))))+1] = Float32(Int32(0))
     end
    end
    ((si_c8 += Int32(1))-Int32(1))
   end
   ((sj_c7 += Int32(1))-Int32(1))
  end
 end
 nothing
end

function STSG_filter_body!(img_NDVI,img_QA,reference_data,StartY,TotalY,Buffer_Up,Buffer_Dn,n_X,n_Y,n_B,n_Years,win,sampcorr,snow_address,vector_out,d_vector_in,d_res,d_res_3,d_index,bidx::UInt32,bidy::UInt32,bdimx::UInt32,bdimy::UInt32,tidx::UInt32,tidy::UInt32)
 @inbounds begin
  local i_c1::Int32 = Int32(UInt32((UInt32((bidx*bdimx))+tidx)))
  if (i_c1>=n_X)
   return nothing
  end
  local j_c2::Int32 = Int32(UInt32((UInt32((bidy*bdimy))+tidy)))
  if (j_c2>=n_Y)
   return nothing
  end
  local samp_c3::Int32 = Int32(Int32(0))
  local aap_c4::Int32 = Int32(Int32(0))
  local similar_index_c5 = Offset(d_index,Int32(Int32((Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(3)))))
  local slope_intercept_c6 = Offset(d_res_3,Int32(Int32((Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(3)))))
  local corr_similar_c7 = Offset(d_res_3,Int32(Int32((Int32((Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(3)))+Int32((Int32((Int32((Int32((Int32(2)*win))+Int32(1)))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(2)))))))
  local y_c8::Int32 = Int32(Int32(0))
  while (y_c8<n_Years)
   local vector_in_c9 = Offset(d_vector_in,Int32(Int32((Int32((i_c1+Int32((j_c2*n_X))))*n_B))))
   local k_c10::Int32 = Int32(Int32(0))
   while (k_c10<n_B)
    vector_in_c9[Int(k_c10)+1] = Float32(img_NDVI[Int(Int32((Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c10*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))))+Int32((Int32((Int32((y_c8*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))*n_B)))))+1])
    ((k_c10 += Int32(1))-Int32(1))
   end
   local vector_in_max_1_c11::Float32 = Float32(vector_in_c9[Int(Int32(0))+1])
   local vector_in_max_2_c12::Float32 = Float32(vector_in_c9[Int(Int32(1))+1])
   local vector_in_max_3_c13::Float32 = Float32(vector_in_c9[Int(Int32(2))+1])
   if (vector_in_max_2_c12>=vector_in_max_1_c11)
    vector_in_max_2_c12 = Float32(vector_in_max_1_c11)
    vector_in_max_1_c11 = Float32(vector_in_c9[Int(Int32(1))+1])
   end
   if (vector_in_max_3_c13>=vector_in_max_1_c11)
    vector_in_max_3_c13 = Float32(vector_in_max_2_c12)
    vector_in_max_2_c12 = Float32(vector_in_max_1_c11)
    vector_in_max_1_c11 = Float32(vector_in_c9[Int(Int32(2))+1])
   else
    if ((vector_in_max_3_c13>=vector_in_max_2_c12)&&(vector_in_max_3_c13<vector_in_max_1_c11))
     vector_in_max_3_c13 = Float32(vector_in_max_2_c12)
     vector_in_max_2_c12 = Float32(vector_in_c9[Int(Int32(2))+1])
    end
   end
   local k_c14::Int32 = Int32(Int32(3))
   while (k_c14<n_B)
    if (vector_in_c9[Int(k_c14)+1]>=vector_in_max_1_c11)
     vector_in_max_3_c13 = Float32(vector_in_max_2_c12)
     vector_in_max_2_c12 = Float32(vector_in_max_1_c11)
     vector_in_max_1_c11 = Float32(vector_in_c9[Int(k_c14)+1])
    else
     if ((vector_in_c9[Int(k_c14)+1]>=vector_in_max_2_c12)&&(vector_in_c9[Int(k_c14)+1]<vector_in_max_1_c11))
      vector_in_max_3_c13 = Float32(vector_in_max_2_c12)
      vector_in_max_2_c12 = Float32(vector_in_c9[Int(k_c14)+1])
     else
      if ((vector_in_c9[Int(k_c14)+1]>=vector_in_max_3_c13)&&(vector_in_c9[Int(k_c14)+1]<vector_in_max_2_c12))
       vector_in_max_3_c13 = Float32(vector_in_c9[Int(k_c14)+1])
      end
     end
    end
    ((k_c14 += Int32(1))-Int32(1))
   end
   if (Float32((Float32((Float32((vector_in_max_1_c11+vector_in_max_2_c12))+vector_in_max_3_c13))/Int32(3)))>Float32(0.15))
    local indic_c15::Int32 = Int32(Int32(0))
    if (y_c8==Int32(0))
     local corr_res_c16 = Offset(d_res,Int32(Int32((Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(4)))))
     local Slope_res_c17 = Offset(d_res,Int32(Int32((Int32((Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(4)))+Int32((Int32((Int32((Int32(2)*win))+Int32(1)))*Int32((Int32((Int32(2)*win))+Int32(1)))))))))
     local Intercept_res_c18 = Offset(d_res,Int32(Int32((Int32((Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(4)))+Int32((Int32((Int32((Int32((Int32(2)*win))+Int32(1)))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(2)))))))
     local new_corr_similar_res_c19 = Offset(d_res,Int32(Int32((Int32((Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(4)))+Int32((Int32((Int32((Int32((Int32(2)*win))+Int32(1)))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(3)))))))
     local count_corr_Slope_c20::Int32 = Int32(Int32(0))
     local m_c21::Int32 = Int32(Int32(0))
     while (m_c21<Int32((Int32((Int32((Int32(2)*win))+Int32(1)))*Int32((Int32((Int32(2)*win))+Int32(1))))))
      if ((corr_res_c16[Int(m_c21)+1]>=sampcorr)&&(Slope_res_c17[Int(m_c21)+1]!=Float32(0.)))
       ((count_corr_Slope_c20 += Int32(1))-Int32(1))
      end
      ((m_c21 += Int32(1))-Int32(1))
     end
     if (count_corr_Slope_c20>=Int32(2))
      samp_c3 = Int32(Int32((count_corr_Slope_c20-Int32(1))))
      local new_corr_c22 = Offset(d_index,Int32(Int32((Int32((Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(3)))+Int32((Int32((Int32((Int32((Int32(2)*win))+Int32(1)))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(2)))))))
      local m_c23::Int32 = Int32(Int32(0))
      while (m_c23<Int32((Int32((Int32((Int32(2)*win))+Int32(1)))*Int32((Int32((Int32(2)*win))+Int32(1))))))
       new_corr_c22[Int(m_c23)+1] = Int32(m_c23)
       ((m_c23 += Int32(1))-Int32(1))
      end
      local m_c24::Int32 = Int32(Int32(0))
      while (m_c24<count_corr_Slope_c20)
       local n_c25::Int32 = Int32(Int32((m_c24+Int32(1))))
       while (n_c25<Int32((Int32((Int32((Int32(2)*win))+Int32(1)))*Int32((Int32((Int32(2)*win))+Int32(1))))))
        if (corr_res_c16[Int(new_corr_c22[Int(m_c24)+1])+1]<corr_res_c16[Int(new_corr_c22[Int(n_c25)+1])+1])
         local temp_c26::Int32 = Int32(new_corr_c22[Int(m_c24)+1])
         new_corr_c22[Int(m_c24)+1] = Int32(new_corr_c22[Int(n_c25)+1])
         new_corr_c22[Int(n_c25)+1] = Int32(temp_c26)
        end
        ((n_c25 += Int32(1))-Int32(1))
       end
       ((m_c24 += Int32(1))-Int32(1))
      end
      local k_c27::Int32 = Int32(Int32(0))
      while (k_c27<samp_c3)
       similar_index_c5[Int(Int32((Int32(1)+Int32((k_c27*Int32(2))))))+1] = Int32(Int32((Int32((Int32(Int32(div(new_corr_c22[Int(Int32((k_c27+Int32(1))))+1],Int32((Int32((Int32(2)*win))+Int32(1))))))+j_c2))-win)))
       similar_index_c5[Int(Int32((k_c27*Int32(2))))+1] = Int32(Int32((Int32((Int32((new_corr_c22[Int(Int32((k_c27+Int32(1))))+1]-Int32((Int32(Int32(div(new_corr_c22[Int(Int32((k_c27+Int32(1))))+1],Int32((Int32((Int32(2)*win))+Int32(1))))))*Int32((Int32((Int32(2)*win))+Int32(1)))))))+i_c1))-win)))
       slope_intercept_c6[Int(Int32((Int32(1)+Int32((k_c27*Int32(2))))))+1] = Float32(Slope_res_c17[Int(Int32((Int32(Int32(div(new_corr_c22[Int(Int32((k_c27+Int32(1))))+1],Int32((Int32((Int32(2)*win))+Int32(1))))))+Int32((Int32((Int32((Int32(2)*win))+Int32(1)))*Int32((new_corr_c22[Int(Int32((k_c27+Int32(1))))+1]-Int32((Int32(Int32(div(new_corr_c22[Int(Int32((k_c27+Int32(1))))+1],Int32((Int32((Int32(2)*win))+Int32(1))))))*Int32((Int32((Int32(2)*win))+Int32(1))))))))))))+1])
       slope_intercept_c6[Int(Int32((k_c27*Int32(2))))+1] = Float32(Intercept_res_c18[Int(Int32((Int32(Int32(div(new_corr_c22[Int(Int32((k_c27+Int32(1))))+1],Int32((Int32((Int32(2)*win))+Int32(1))))))+Int32((Int32((Int32((Int32(2)*win))+Int32(1)))*Int32((new_corr_c22[Int(Int32((k_c27+Int32(1))))+1]-Int32((Int32(Int32(div(new_corr_c22[Int(Int32((k_c27+Int32(1))))+1],Int32((Int32((Int32(2)*win))+Int32(1))))))*Int32((Int32((Int32(2)*win))+Int32(1))))))))))))+1])
       corr_similar_c7[Int(k_c27)+1] = Float32(new_corr_similar_res_c19[Int(Int32((Int32(Int32(div(new_corr_c22[Int(Int32((k_c27+Int32(1))))+1],Int32((Int32((Int32(2)*win))+Int32(1))))))+Int32((Int32((Int32((Int32(2)*win))+Int32(1)))*Int32((new_corr_c22[Int(Int32((k_c27+Int32(1))))+1]-Int32((Int32(Int32(div(new_corr_c22[Int(Int32((k_c27+Int32(1))))+1],Int32((Int32((Int32(2)*win))+Int32(1))))))*Int32((Int32((Int32(2)*win))+Int32(1))))))))))))+1])
       ((k_c27 += Int32(1))-Int32(1))
      end
      aap_c4 = Int32(Int32(1))
     else
      aap_c4 = Int32(Int32(0))
     end
    end
    local trend_NDVI_c28 = Offset(d_res,Int32(Int32((Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(4)))))
    if (aap_c4==Int32(1))
     local count_trend_NDVI_c29::Int32 = Int32(Int32(0))
     local nocount_trend_NDVI_c30::Int32 = Int32(Int32(0))
     local res_trend_NDVI_c31 = Offset(d_index,Int32(Int32((Int32((Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(3)))+Int32((Int32((Int32((Int32((Int32(2)*win))+Int32(1)))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(2)))))))
     local nores_trend_NDVI_c32 = Offset(res_trend_NDVI_c31,Int32(n_B))
     local count_conres_c33::Int32 = Int32(Int32(0))
     local k_c34::Int32 = Int32(Int32(0))
     while (k_c34<n_B)
      local temp_NDVI_c35 = Offset(d_res,Int32(Int32((Int32((Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(4)))+Int32((Int32((Int32((Int32(2)*win))+Int32(1)))*Int32((Int32((Int32(2)*win))+Int32(1)))))))))
      local count_temp_NDVI_c36::Int32 = Int32(Int32(0))
      local total_new_corr_similar_c37::Float32 = Float32(Int32(0))
      local total_new_temp_c38::Float32 = Float32(Int32(0))
      local m_c39::Int32 = Int32(Int32(0))
      while (m_c39<samp_c3)
       if ((((((similar_index_c5[Int(Int32((m_c39*Int32(2))))+1]>=Int32(0))&&(similar_index_c5[Int(Int32((m_c39*Int32(2))))+1]<n_X))&&(Int32((similar_index_c5[Int(Int32((Int32((m_c39*Int32(2)))+Int32(1))))+1]+Buffer_Up))>=Int32(0)))&&(Int32((similar_index_c5[Int(Int32((Int32((m_c39*Int32(2)))+Int32(1))))+1]+Buffer_Up))<TotalY))&&((img_QA[Int(Int32((Int32((Int32((similar_index_c5[Int(Int32((m_c39*Int32(2))))+1]+Int32((Int32((similar_index_c5[Int(Int32((Int32(1)+Int32((m_c39*Int32(2))))))+1]+Buffer_Up))*n_X))))+Int32((Int32((k_c34*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))))+Int32((Int32((Int32((y_c8*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))*n_B)))))+1]==Float32(0.))||(img_QA[Int(Int32((Int32((Int32((similar_index_c5[Int(Int32((m_c39*Int32(2))))+1]+Int32((Int32((similar_index_c5[Int(Int32((Int32(1)+Int32((m_c39*Int32(2))))))+1]+Buffer_Up))*n_X))))+Int32((Int32((k_c34*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))))+Int32((Int32((Int32((y_c8*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))*n_B)))))+1]==Float32(1.))))&&(reference_data[Int(Int32((Int32((similar_index_c5[Int(Int32((m_c39*Int32(2))))+1]+Int32((Int32((similar_index_c5[Int(Int32((Int32(1)+Int32((m_c39*Int32(2))))))+1]+Buffer_Up))*n_X))))+Int32((Int32((k_c34*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn)))))))+1]!=Float32(0.)))
        local new_ratio_c40::Float32 = Float32(Float32((img_NDVI[Int(Int32((Int32((Int32((similar_index_c5[Int(Int32((m_c39*Int32(2))))+1]+Int32((Int32((similar_index_c5[Int(Int32((Int32(1)+Int32((m_c39*Int32(2))))))+1]+Buffer_Up))*n_X))))+Int32((Int32((k_c34*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))))+Int32((Int32((Int32((y_c8*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))*n_B)))))+1]/reference_data[Int(Int32((Int32((similar_index_c5[Int(Int32((m_c39*Int32(2))))+1]+Int32((Int32((similar_index_c5[Int(Int32((Int32(1)+Int32((m_c39*Int32(2))))))+1]+Buffer_Up))*n_X))))+Int32((Int32((k_c34*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn)))))))+1])))
        temp_NDVI_c35[Int(m_c39)+1] = Float32(Float32((muladd(Float32(new_ratio_c40),Float32(slope_intercept_c6[Int(Int32((Int32(1)+Int32((m_c39*Int32(2))))))+1]),Float32(slope_intercept_c6[Int(Int32((m_c39*Int32(2))))+1]))*reference_data[Int(Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c34*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn)))))))+1])))
        if ((temp_NDVI_c35[Int(m_c39)+1]>=Float32(1.))||(temp_NDVI_c35[Int(m_c39)+1]<=Float32((-Float32(0.2)))))
         temp_NDVI_c35[Int(m_c39)+1] = Float32(Float32(0.))
        end
        if (temp_NDVI_c35[Int(m_c39)+1]!=Float32(0.))
         ((count_temp_NDVI_c36 += Int32(1))-Int32(1))
         total_new_corr_similar_c37 += corr_similar_c7[Int(m_c39)+1]
        end
       else
        temp_NDVI_c35[Int(m_c39)+1] = Float32(Float32(0.))
       end
       ((m_c39 += Int32(1))-Int32(1))
      end
      if (count_temp_NDVI_c36!=Int32(0))
       local m_c41::Int32 = Int32(Int32(0))
       while (m_c41<samp_c3)
        if (temp_NDVI_c35[Int(m_c41)+1]!=Int32(0))
         total_new_temp_c38 = Float32(muladd(Float32(Float32((corr_similar_c7[Int(m_c41)+1]/total_new_corr_similar_c37))),Float32(temp_NDVI_c35[Int(m_c41)+1]),Float32(total_new_temp_c38)))
        end
        ((m_c41 += Int32(1))-Int32(1))
       end
       trend_NDVI_c28[Int(k_c34)+1] = Float32(total_new_temp_c38)
       if (trend_NDVI_c28[Int(k_c34)+1]!=Int32(0))
        res_trend_NDVI_c31[Int(((count_trend_NDVI_c29 += Int32(1))-Int32(1)))+1] = Int32(k_c34)
        if ((count_trend_NDVI_c29>Int32(1))&&(Int32((k_c34-res_trend_NDVI_c31[Int(Int32((count_trend_NDVI_c29-Int32(2))))+1]))>=Int32(3)))
         ((count_conres_c33 += Int32(1))-Int32(1))
        end
       else
        nores_trend_NDVI_c32[Int(((nocount_trend_NDVI_c30 += Int32(1))-Int32(1)))+1] = Int32(k_c34)
       end
      else
       trend_NDVI_c28[Int(k_c34)+1] = Float32(Int32(0))
       nores_trend_NDVI_c32[Int(((nocount_trend_NDVI_c30 += Int32(1))-Int32(1)))+1] = Int32(k_c34)
      end
      ((k_c34 += Int32(1))-Int32(1))
     end
     if ((count_trend_NDVI_c29>=Int32(div(n_B,Int32(2))))&&(count_conres_c33==Int32(0)))
      local m_c42::Int32 = Int32(Int32(0))
      while (m_c42<nocount_trend_NDVI_c30)
       local sta_c43::Int32 = Int32(Int32(0))
       if (nores_trend_NDVI_c32[Int(m_c42)+1]<res_trend_NDVI_c31[Int(Int32(0))+1])
        sta_c43 = Int32(Int32(0))
       else
        if (nores_trend_NDVI_c32[Int(m_c42)+1]>res_trend_NDVI_c31[Int(Int32((count_trend_NDVI_c29-Int32(1))))+1])
         sta_c43 = Int32(Int32((count_trend_NDVI_c29-Int32(4))))
        else
         local n_c44::Int32 = Int32(Int32(0))
         while (n_c44<Int32((count_trend_NDVI_c29-Int32(1))))
          if ((res_trend_NDVI_c31[Int(n_c44)+1]<nores_trend_NDVI_c32[Int(m_c42)+1])&&(nores_trend_NDVI_c32[Int(m_c42)+1]<res_trend_NDVI_c31[Int(Int32((n_c44+Int32(1))))+1]))
           if (Int32((n_c44-Int32(1)))<Int32(0))
            sta_c43 = Int32(Int32(0))
           else
            if (Int32((count_trend_NDVI_c29-n_c44))<Int32(4))
             sta_c43 = Int32(Int32((count_trend_NDVI_c29-Int32(4))))
            else
             sta_c43 = Int32(Int32((n_c44-Int32(1))))
            end
           end
           break
          end
          ((n_c44 += Int32(1))-Int32(1))
         end
        end
       end
       local x_c45 = MVector{4,Float32}((Float32(NaN),Float32(NaN),Float32(NaN),Float32(NaN),))
       local y_c46 = MVector{4,Float32}((Float32(NaN),Float32(NaN),Float32(NaN),Float32(NaN),))
       local n_c47::Int32 = Int32(Int32(0))
       while (n_c47<Int32(4))
        x_c45[Int(n_c47)+1] = Float32(res_trend_NDVI_c31[Int(Int32((sta_c43+n_c47)))+1])
        y_c46[Int(n_c47)+1] = Float32(trend_NDVI_c28[Int(res_trend_NDVI_c31[Int(Int32((sta_c43+n_c47)))+1])+1])
        ((n_c47 += Int32(1))-Int32(1))
       end
       local sig_c48::Float32 = Float32(NaN)
       local p_c49::Float32 = Float32(NaN)
       local y2_c50 = MVector{4,Float32}((Int32(0),Float32(0),Float32(0),Float32(0),))
       local u_c51 = MVector{4,Float32}((Int32(0),Float32(0),Float32(0),Float32(0),))
       local n_c52::Int32 = Int32(Int32(1))
       while (n_c52<Int32(3))
        sig_c48 = Float32(Float32((Float32((x_c45[Int(n_c52)+1]-x_c45[Int(Int32((n_c52-Int32(1))))+1]))/Float32((x_c45[Int(Int32((n_c52+Int32(1))))+1]-x_c45[Int(Int32((n_c52-Int32(1))))+1])))))
        p_c49 = Float32(muladd(Float32(sig_c48),Float32(y2_c50[Int(Int32((n_c52-Int32(1))))+1]),Float32(Float32(2.))))
        y2_c50[Int(n_c52)+1] = Float32(Float32((Float32((sig_c48-Int32(1)))/p_c49)))
        u_c51[Int(n_c52)+1] = Float32(Float32((Float32((Float32((y_c46[Int(Int32((n_c52+Int32(1))))+1]-y_c46[Int(n_c52)+1]))/Float32((x_c45[Int(Int32((n_c52+Int32(1))))+1]-x_c45[Int(n_c52)+1]))))-Float32((Float32((y_c46[Int(n_c52)+1]-y_c46[Int(Int32((n_c52-Int32(1))))+1]))/Float32((x_c45[Int(n_c52)+1]-x_c45[Int(Int32((n_c52-Int32(1))))+1])))))))
        u_c51[Int(n_c52)+1] = Float32(Float32((muladd((-Float32(sig_c48)),Float32(u_c51[Int(Int32((n_c52-Int32(1))))+1]),Float32(Float32((Float32((Float32(6.)*u_c51[Int(n_c52)+1]))/Float32((x_c45[Int(Int32((n_c52+Int32(1))))+1]-x_c45[Int(Int32((n_c52-Int32(1))))+1]))))))/p_c49)))
        ((n_c52 += Int32(1))-Int32(1))
       end
       local n_c53::Int32 = Int32(Int32(2))
       while (n_c53>=Int32(0))
        y2_c50[Int(n_c53)+1] = Float32(muladd(Float32(y2_c50[Int(n_c53)+1]),Float32(y2_c50[Int(Int32((n_c53+Int32(1))))+1]),Float32(u_c51[Int(n_c53)+1])))
        ((n_c53 -= Int32(1))+Int32(1))
       end
       local klo_c54::Int32 = Int32(Int32(0))
       local khi_c55::Int32 = Int32(Int32(3))
       while (Int32((khi_c55-klo_c54))>Int32(1))
        local k_c56::Int32 = Int32(Int32((Int32((khi_c55+klo_c54))>>Int32(1))))
        if (x_c45[Int(k_c56)+1]>nores_trend_NDVI_c32[Int(m_c42)+1])
         khi_c55 = Int32(k_c56)
        else
         klo_c54 = Int32(k_c56)
        end
       end
       local h_c57::Float32 = Float32(Float32((x_c45[Int(khi_c55)+1]-x_c45[Int(klo_c54)+1])))
       local a_c58::Float32 = Float32(Float32((Float32((x_c45[Int(khi_c55)+1]-nores_trend_NDVI_c32[Int(m_c42)+1]))/h_c57)))
       local b_c59::Float32 = Float32(Float32((Float32((nores_trend_NDVI_c32[Int(m_c42)+1]-x_c45[Int(klo_c54)+1]))/h_c57)))
       trend_NDVI_c28[Int(nores_trend_NDVI_c32[Int(m_c42)+1])+1] = Float32(Float32((muladd(Float32(a_c58),Float32(y_c46[Int(klo_c54)+1]),Float32(Float32((b_c59*y_c46[Int(khi_c55)+1]))))+Float32((Float32((Float32((muladd(Float32(muladd(Float32(Float32((a_c58*a_c58))),Float32(a_c58),(-Float32(a_c58)))),Float32(y2_c50[Int(klo_c54)+1]),Float32(Float32((muladd(Float32(Float32((b_c59*b_c59))),Float32(b_c59),(-Float32(b_c59)))*y2_c50[Int(khi_c55)+1]))))*h_c57))*h_c57))/Float32(6.))))))
       ((m_c42 += Int32(1))-Int32(1))
      end
      indic_c15 = Int32(Int32(1))
     else
      indic_c15 = Int32(Int32(0))
     end
    else
     indic_c15 = Int32(Int32(0))
    end
    if (indic_c15==Int32(1))
     if (snow_address==Int32(1))
      local count_vector_QA_c60::Int32 = Int32(Int32(0))
      local k_c61::Int32 = Int32(Int32(0))
      while (k_c61<n_B)
       if (img_QA[Int(Int32((Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c61*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))))+Int32((Int32((Int32((y_c8*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))*n_B)))))+1]==Float32(2.))
        ((count_vector_QA_c60 += Int32(1))-Int32(1))
       end
       ((k_c61 += Int32(1))-Int32(1))
      end
      if (count_vector_QA_c60!=Int32(0))
       local bv_count_c62::Int32 = Int32(Int32(0))
       local bv_total_c63::Float32 = Float32(Float64(0.))
       local yeari_c64::Int32 = Int32(Int32(0))
       while (yeari_c64<n_Years)
        local k_c65::Int32 = Int32(Int32(0))
        while (k_c65<Int32(6))
         if ((img_QA[Int(Int32((Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c65*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))))+Int32((Int32((Int32((yeari_c64*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))*n_B)))))+1]==Float32(0.))||(img_QA[Int(Int32((Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c65*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))))+Int32((Int32((Int32((yeari_c64*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))*n_B)))))+1]==Float32(1.)))
          ((bv_count_c62 += Int32(1))-Int32(1))
          bv_total_c63 = Float32(Float32((bv_total_c63+img_NDVI[Int(Int32((Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c65*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))))+Int32((Int32((Int32((yeari_c64*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))*n_B)))))+1])))
         end
         ((k_c65 += Int32(1))-Int32(1))
        end
        ((yeari_c64 += Int32(1))-Int32(1))
       end
       if (bv_count_c62!=Int32(0))
        local bv_c66::Float32 = Float32(Float32((bv_total_c63/bv_count_c62)))
        local k_c67::Int32 = Int32(Int32(0))
        while (k_c67<n_B)
         if (img_QA[Int(Int32((Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c67*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))))+Int32((Int32((Int32((y_c8*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))*n_B)))))+1]==Float32(2.))
          vector_in_c9[Int(k_c67)+1] = Float32(bv_c66)
          trend_NDVI_c28[Int(k_c67)+1] = Float32(bv_c66)
         end
         ((k_c67 += Int32(1))-Int32(1))
        end
       end
      end
     end
     local gdis_c68::Float32 = Float32(Float32(0.))
     local fl_c69 = Offset(d_res,Int32(Int32((Int32((Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(4)))+Int32((Int32((Int32((Int32((Int32(2)*win))+Int32(1)))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(2)))))))
     local count_fl_c70::Int32 = Int32(Int32(0))
     local mean_fl_c71::Float32 = Float32(Int32(0))
     local k_c72::Int32 = Int32(Int32(0))
     while (k_c72<n_B)
      if ((img_QA[Int(Int32((Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c72*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))))+Int32((Int32((Int32((y_c8*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))*n_B)))))+1]==Float32(0.))||(img_QA[Int(Int32((Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c72*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))))+Int32((Int32((Int32((y_c8*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))*n_B)))))+1]==Float32(1.)))
       fl_c69[Int(k_c72)+1] = Float32(Float32((vector_in_c9[Int(k_c72)+1]-trend_NDVI_c28[Int(k_c72)+1])))
       ((count_fl_c70 += Int32(1))-Int32(1))
       mean_fl_c71 += fl_c69[Int(k_c72)+1]
      else
       fl_c69[Int(k_c72)+1] = Float32(Float32((-Float32(1.))))
      end
      ((k_c72 += Int32(1))-Int32(1))
     end
     if (count_fl_c70!=Int32(0))
      mean_fl_c71 = Float32(Float32((mean_fl_c71/count_fl_c70)))
     end
     local k_c73::Int32 = Int32(Int32(0))
     while (k_c73<n_B)
      if (fl_c69[Int(k_c73)+1]==Float32((-Float32(1.))))
       fl_c69[Int(k_c73)+1] = Float32(mean_fl_c71)
      end
      ((k_c73 += Int32(1))-Int32(1))
     end
     local k_c74::Int32 = Int32(Int32(0))
     while (k_c74<n_B)
      local min_fl_c75::Float32 = Float32(Int32(0))
      local max_fl_c76::Float32 = Float32(Int32(0))
      local k_c77::Int32 = Int32(Int32(0))
      while (k_c77<n_B)
       if (min_fl_c75>fl_c69[Int(k_c77)+1])
        min_fl_c75 = Float32(fl_c69[Int(k_c77)+1])
       end
       if (max_fl_c76<fl_c69[Int(k_c77)+1])
        max_fl_c76 = Float32(fl_c69[Int(k_c77)+1])
       end
       ((k_c77 += Int32(1))-Int32(1))
      end
      fl_c69[Int(k_c74)+1] = Float32(Float32((Float32((fl_c69[Int(k_c74)+1]-min_fl_c75))/Float32((max_fl_c76-min_fl_c75)))))
      if (Float32((vector_in_c9[Int(k_c74)+1]-trend_NDVI_c28[Int(k_c74)+1]))>=Float32(0.))
       gdis_c68 = Float32(muladd(Float32(fl_c69[Int(k_c74)+1]),Float32(Float32((vector_in_c9[Int(k_c74)+1]-trend_NDVI_c28[Int(k_c74)+1]))),Float32(gdis_c68)))
      else
       gdis_c68 = Float32(muladd(Float32(fl_c69[Int(k_c74)+1]),Float32(Float32((trend_NDVI_c28[Int(k_c74)+1]-vector_in_c9[Int(k_c74)+1]))),Float32(gdis_c68)))
      end
      ((k_c74 += Int32(1))-Int32(1))
     end
     local k_c78::Int32 = Int32(Int32(0))
     while (k_c78<n_B)
      if (img_QA[Int(Int32((Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c78*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))))+Int32((Int32((Int32((y_c8*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))*n_B)))))+1]==Float32(0.))
       trend_NDVI_c28[Int(k_c78)+1] = Float32(vector_in_c9[Int(k_c78)+1])
      end
      if ((img_QA[Int(Int32((Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c78*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))))+Int32((Int32((Int32((y_c8*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))*n_B)))))+1]!=Float32(0.))&&(img_QA[Int(Int32((Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c78*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))))+Int32((Int32((Int32((y_c8*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))*n_B)))))+1]!=Float32(1.)))
       vector_in_c9[Int(k_c78)+1] = Float32(trend_NDVI_c28[Int(k_c78)+1])
      end
      ((k_c78 += Int32(1))-Int32(1))
     end
     local vec_fil_c79 = Offset(d_res,Int32(Int32((Int32((Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(4)))+Int32((Int32((Int32((Int32((Int32(2)*win))+Int32(1)))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(3)))))))
     local ormax_c80::Float32 = Float32(gdis_c68)
     local loop_times_c81::Int32 = Int32(Int32(1))
     while ((gdis_c68<=ormax_c80)&&(loop_times_c81<Int32(50)))
      loop_times_c81 = Int32(Int32((loop_times_c81+Int32(1))))
      local k_c82::Int32 = Int32(Int32(0))
      while (k_c82<n_B)
       vec_fil_c79[Int(k_c82)+1] = Float32(trend_NDVI_c28[Int(k_c82)+1])
       ((k_c82 += Int32(1))-Int32(1))
      end
      local savgolFilter_c83 = MVector{9,Float64}((Float64((-Float64(0.00543880))),Float64(0.0435097),Float64((-Float64(0.152289))),Float64(0.304585),Float64(0.619267),Float64(0.304585),Float64((-Float64(0.152289))),Float64(0.0435097),Float64((-Float64(0.00543880))),))
      local savgolFilterW_c84::Int32 = Int32(Int32(div(Int32(72),Int32(8))))
      local ra4W_c85::Int32 = Int32(n_B)
      local new_ra4_c86 = Offset(d_res,Int32(Int32((Int32((Int32((Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(4)))+Int32((Int32((Int32((Int32((Int32(2)*win))+Int32(1)))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(3)))))+n_B))))
      local ii_c87::Int32 = Int32(Int32(0))
      while (ii_c87<Int32((Int32((savgolFilterW_c84+ra4W_c85))-Int32(1))))
       if (ii_c87<Int32(div(Int32((savgolFilterW_c84-Int32(1))),Int32(2))))
        new_ra4_c86[Int(ii_c87)+1] = Float32(Float32(((vector_in_c9[Int(Int32(0))+1]>=trend_NDVI_c28[Int(Int32(0))+1]) ? vector_in_c9[Int(Int32(0))+1] : trend_NDVI_c28[Int(Int32(0))+1])))
       else
        if (ii_c87>Int32((Int32((Int32(div(Int32((savgolFilterW_c84-Int32(1))),Int32(2)))+ra4W_c85))-Int32(1))))
         new_ra4_c86[Int(ii_c87)+1] = Float32(Float32(((vector_in_c9[Int(Int32((ra4W_c85-Int32(1))))+1]>=trend_NDVI_c28[Int(Int32((ra4W_c85-Int32(1))))+1]) ? vector_in_c9[Int(Int32((ra4W_c85-Int32(1))))+1] : trend_NDVI_c28[Int(Int32((ra4W_c85-Int32(1))))+1])))
        else
         new_ra4_c86[Int(ii_c87)+1] = Float32(Float32(((vector_in_c9[Int(Int32((ii_c87-Int32(div(Int32((savgolFilterW_c84-Int32(1))),Int32(2))))))+1]>=trend_NDVI_c28[Int(Int32((ii_c87-Int32(div(Int32((savgolFilterW_c84-Int32(1))),Int32(2))))))+1]) ? vector_in_c9[Int(Int32((ii_c87-Int32(div(Int32((savgolFilterW_c84-Int32(1))),Int32(2))))))+1] : trend_NDVI_c28[Int(Int32((ii_c87-Int32(div(Int32((savgolFilterW_c84-Int32(1))),Int32(2))))))+1])))
        end
       end
       ((ii_c87 += Int32(1))-Int32(1))
      end
      local ii_c88::Int32 = Int32(Int32(0))
      while (ii_c88<ra4W_c85)
       local temp_c89::Float32 = Float32(Int32(0))
       local jj_c90::Int32 = Int32(Int32(0))
       while (jj_c90<savgolFilterW_c84)
        temp_c89 = Float32(muladd(Float64(savgolFilter_c83[Int(jj_c90)+1]),Float64(new_ra4_c86[Int(Int32((ii_c88+jj_c90)))+1]),Float64(temp_c89)))
        ((jj_c90 += Int32(1))-Int32(1))
       end
       trend_NDVI_c28[Int(ii_c88)+1] = Float32(temp_c89)
       ((ii_c88 += Int32(1))-Int32(1))
      end
      ormax_c80 = Float32(gdis_c68)
      gdis_c68 = Float32(Float32(0.))
      local k_c91::Int32 = Int32(Int32(0))
      while (k_c91<n_B)
       if (Float32((vector_in_c9[Int(k_c91)+1]-trend_NDVI_c28[Int(k_c91)+1]))>=Int32(0))
        gdis_c68 = Float32(muladd(Float32(fl_c69[Int(k_c91)+1]),Float32(Float32((vector_in_c9[Int(k_c91)+1]-trend_NDVI_c28[Int(k_c91)+1]))),Float32(gdis_c68)))
       else
        gdis_c68 = Float32(muladd(Float32(fl_c69[Int(k_c91)+1]),Float32(Float32((trend_NDVI_c28[Int(k_c91)+1]-vector_in_c9[Int(k_c91)+1]))),Float32(gdis_c68)))
       end
       ((k_c91 += Int32(1))-Int32(1))
      end
     end
     local k_c92::Int32 = Int32(Int32(0))
     while (k_c92<n_B)
      vector_out[Int(Int32((Int32((Int32((i_c1+Int32((Int32((j_c2+StartY))*n_X))))+Int32((Int32((k_c92*n_X))*TotalY))))+Int32((Int32((Int32((y_c8*n_X))*TotalY))*n_B)))))+1] = Float32(vec_fil_c79[Int(k_c92)+1])
      ((k_c92 += Int32(1))-Int32(1))
     end
     local smi_c93::Int32 = Int32(Int32(0))
     while (smi_c93<Int32((n_B-Int32(4))))
      local a1_c94::Float32 = Float32(vec_fil_c79[Int(smi_c93)+1])
      local a2_c95::Float32 = Float32(vec_fil_c79[Int(Int32((smi_c93+Int32(1))))+1])
      local a3_c96::Float32 = Float32(vec_fil_c79[Int(Int32((smi_c93+Int32(2))))+1])
      local a4_c97::Float32 = Float32(vec_fil_c79[Int(Int32((smi_c93+Int32(3))))+1])
      local a5_c98::Float32 = Float32(vec_fil_c79[Int(Int32((smi_c93+Int32(4))))+1])
      if ((((a1_c94>a2_c95)&&(a2_c95<a3_c96))&&(a3_c96>a4_c97))&&(a4_c97<a5_c98))
       vector_out[Int(Int32((Int32((Int32((i_c1+Int32((Int32((j_c2+StartY))*n_X))))+Int32((Int32((Int32((smi_c93+Int32(1)))*n_X))*TotalY))))+Int32((Int32((Int32((y_c8*n_X))*TotalY))*n_B)))))+1] = Float32(Float32((Float32((a1_c94+a3_c96))/Float32(2.))))
       vector_out[Int(Int32((Int32((Int32((i_c1+Int32((Int32((j_c2+StartY))*n_X))))+Int32((Int32((Int32((smi_c93+Int32(3)))*n_X))*TotalY))))+Int32((Int32((Int32((y_c8*n_X))*TotalY))*n_B)))))+1] = Float32(Float32((Float32((a3_c96+a5_c98))/Float32(2.))))
      end
      ((smi_c93 += Int32(1))-Int32(1))
     end
    end
    if (indic_c15==Int32(0))
     if (snow_address==Int32(1))
      local count_vector_QA_c99::Int32 = Int32(Int32(0))
      local k_c100::Int32 = Int32(Int32(0))
      while (k_c100<n_B)
       if (img_QA[Int(Int32((Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c100*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))))+Int32((Int32((Int32((y_c8*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))*n_B)))))+1]==Float32(2.))
        ((count_vector_QA_c99 += Int32(1))-Int32(1))
       end
       ((k_c100 += Int32(1))-Int32(1))
      end
      if (count_vector_QA_c99!=Int32(0))
       local bv_count_c101::Int32 = Int32(Int32(0))
       local bv_total_c102::Float32 = Float32(Float64(0.))
       local yeari_c103::Int32 = Int32(Int32(0))
       while (yeari_c103<n_Years)
        local k_c104::Int32 = Int32(Int32(0))
        while (k_c104<Int32(6))
         if ((img_QA[Int(Int32((Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c104*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))))+Int32((Int32((Int32((yeari_c103*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))*n_B)))))+1]==Float32(0.))||(img_QA[Int(Int32((Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c104*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))))+Int32((Int32((Int32((yeari_c103*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))*n_B)))))+1]==Float32(1.)))
          ((bv_count_c101 += Int32(1))-Int32(1))
          bv_total_c102 = Float32(Float32((bv_total_c102+img_NDVI[Int(Int32((Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c104*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))))+Int32((Int32((Int32((yeari_c103*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))*n_B)))))+1])))
         end
         ((k_c104 += Int32(1))-Int32(1))
        end
        ((yeari_c103 += Int32(1))-Int32(1))
       end
       if (bv_count_c101!=Int32(0))
        local bv_c105::Float32 = Float32(Float32((bv_total_c102/bv_count_c101)))
        local k_c106::Int32 = Int32(Int32(0))
        while (k_c106<n_B)
         if (img_QA[Int(Int32((Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c106*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))))+Int32((Int32((Int32((y_c8*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))*n_B)))))+1]==Float32(2.))
          vector_in_c9[Int(k_c106)+1] = Float32(bv_c105)
          trend_NDVI_c28[Int(k_c106)+1] = Float32(bv_c105)
         end
         ((k_c106 += Int32(1))-Int32(1))
        end
       end
      end
     end
     local count_vector_QA_c107::Int32 = Int32(Int32(0))
     local res_vector_QA_c108 = Offset(d_index,Int32(Int32((Int32((Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(3)))+Int32((Int32((Int32((Int32((Int32(2)*win))+Int32(1)))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(2)))))))
     local k_c109::Int32 = Int32(Int32(0))
     while (k_c109<n_B)
      if ((img_QA[Int(Int32((Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c109*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))))+Int32((Int32((Int32((y_c8*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))*n_B)))))+1]<=Float32(2.))&&(img_QA[Int(Int32((Int32((Int32((i_c1+Int32((Int32((j_c2+Buffer_Up))*n_X))))+Int32((Int32((k_c109*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))))+Int32((Int32((Int32((y_c8*n_X))*Int32((Int32((n_Y+Buffer_Up))+Buffer_Dn))))*n_B)))))+1]!=Float32((-Float32(1.)))))
       res_vector_QA_c108[Int(((count_vector_QA_c107 += Int32(1))-Int32(1)))+1] = Int32(k_c109)
      end
      ((k_c109 += Int32(1))-Int32(1))
     end
     if ((count_vector_QA_c107<n_B)&&(count_vector_QA_c107>Int32(1)))
      local k_c110::Float64 = Float64(NaN)
      local x_c111::Int32 = Int32(Int32(0))
      local l_c112::Int32 = Int32(Int32(0))
      if (res_vector_QA_c108[Int(Int32((count_vector_QA_c107-Int32(1))))+1]!=Int32((n_B-Int32(1))))
       k_c110 = Float64(Float32((Float32((vector_in_c9[Int(res_vector_QA_c108[Int(Int32((count_vector_QA_c107-Int32(1))))+1])+1]-vector_in_c9[Int(res_vector_QA_c108[Int(Int32((count_vector_QA_c107-Int32(2))))+1])+1]))/Int32((res_vector_QA_c108[Int(Int32((count_vector_QA_c107-Int32(1))))+1]-res_vector_QA_c108[Int(Int32((count_vector_QA_c107-Int32(2))))+1])))))
       vector_in_c9[Int(Int32((n_B-Int32(1))))+1] = Float32(muladd(Float64(k_c110),Float64(Int32((Int32((n_B-Int32(1)))-res_vector_QA_c108[Int(Int32((count_vector_QA_c107-Int32(1))))+1]))),Float64(vector_in_c9[Int(res_vector_QA_c108[Int(Int32((count_vector_QA_c107-Int32(1))))+1])+1])))
       res_vector_QA_c108[Int(((count_vector_QA_c107 += Int32(1))-Int32(1)))+1] = Int32(Int32((n_B-Int32(1))))
      end
      local count_res_vector_QA_c113::Int32 = Int32(count_vector_QA_c107)
      while ((count_vector_QA_c107<n_B)&&(x_c111<Int32((count_res_vector_QA_c113-Int32(1)))))
       l_c112 = Int32(Int32((res_vector_QA_c108[Int(Int32((x_c111+Int32(1))))+1]-res_vector_QA_c108[Int(x_c111)+1])))
       local n_c114::Int32 = Int32(Int32(1))
       while (l_c112>Int32(1))
        k_c110 = Float64(Float32((Float32((vector_in_c9[Int(res_vector_QA_c108[Int(Int32((x_c111+Int32(1))))+1])+1]-vector_in_c9[Int(res_vector_QA_c108[Int(x_c111)+1])+1]))/Int32((res_vector_QA_c108[Int(Int32((x_c111+Int32(1))))+1]-res_vector_QA_c108[Int(x_c111)+1])))))
        vector_in_c9[Int(Int32((res_vector_QA_c108[Int(x_c111)+1]+n_c114)))+1] = Float32(muladd(Float64(k_c110),Float64(n_c114),Float64(vector_in_c9[Int(res_vector_QA_c108[Int(x_c111)+1])+1])))
        ((count_vector_QA_c107 += Int32(1))-Int32(1))
        ((n_c114 += Int32(1))-Int32(1))
        ((l_c112 -= Int32(1))+Int32(1))
       end
       ((x_c111 += Int32(1))-Int32(1))
      end
      if (res_vector_QA_c108[Int(Int32(0))+1]!=Int32(0))
       k_c110 = Float64(Float32((Float32((vector_in_c9[Int(res_vector_QA_c108[Int(Int32(1))+1])+1]-vector_in_c9[Int(res_vector_QA_c108[Int(Int32(0))+1])+1]))/Int32((res_vector_QA_c108[Int(Int32(1))+1]-res_vector_QA_c108[Int(Int32(0))+1])))))
       l_c112 = Int32(res_vector_QA_c108[Int(Int32(0))+1])
       local n_c115::Int32 = Int32(Int32(0))
       while true
        vector_in_c9[Int(n_c115)+1] = Float32(muladd((-Float64(k_c110)),Float64(l_c112),Float64(vector_in_c9[Int(res_vector_QA_c108[Int(Int32(0))+1])+1])))
        ((n_c115 += Int32(1))-Int32(1))
        ((l_c112 -= Int32(1))+Int32(1))
        if !((l_c112>=Int32(1)));break;end
       end
      end
     end
     local rst_c116 = Offset(d_res,Int32(Int32((Int32((Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(4)))+Int32((Int32((Int32((Int32((Int32(2)*win))+Int32(1)))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(3)))))))
     local savgolFilter_c117 = MVector{9,Float64}((Float64((-Float64(0.0909091))),Float64(0.0606061),Float64(0.168831),Float64(0.233766),Float64(0.255411),Float64(0.233766),Float64(0.168831),Float64(0.0606061),Float64((-Float64(0.0909091))),))
     local savgolFilterW_c118::Int32 = Int32(Int32(div(Int32(72),Int32(8))))
     local vector_inW_c119::Int32 = Int32(n_B)
     local new_vector_in_c120 = Offset(d_res,Int32(Int32((Int32((Int32((Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(4)))+Int32((Int32((Int32((Int32((Int32(2)*win))+Int32(1)))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(3)))))+n_B))))
     local ii_c121::Int32 = Int32(Int32(0))
     while (ii_c121<Int32((Int32((savgolFilterW_c118+vector_inW_c119))-Int32(1))))
      if (ii_c121<Int32(div(Int32((savgolFilterW_c118-Int32(1))),Int32(2))))
       new_vector_in_c120[Int(ii_c121)+1] = Float32(vector_in_c9[Int(Int32(0))+1])
      else
       if (ii_c121>Int32((Int32((Int32(div(Int32((savgolFilterW_c118-Int32(1))),Int32(2)))+vector_inW_c119))-Int32(1))))
        new_vector_in_c120[Int(ii_c121)+1] = Float32(vector_in_c9[Int(Int32((vector_inW_c119-Int32(1))))+1])
       else
        new_vector_in_c120[Int(ii_c121)+1] = Float32(vector_in_c9[Int(Int32((ii_c121-Int32(div(Int32((savgolFilterW_c118-Int32(1))),Int32(2))))))+1])
       end
      end
      ((ii_c121 += Int32(1))-Int32(1))
     end
     local ii_c122::Int32 = Int32(Int32(0))
     while (ii_c122<vector_inW_c119)
      local temp_c123::Float32 = Float32(Int32(0))
      local jj_c124::Int32 = Int32(Int32(0))
      while (jj_c124<savgolFilterW_c118)
       temp_c123 = Float32(muladd(Float64(savgolFilter_c117[Int(jj_c124)+1]),Float64(new_vector_in_c120[Int(Int32((ii_c122+jj_c124)))+1]),Float64(temp_c123)))
       ((jj_c124 += Int32(1))-Int32(1))
      end
      rst_c116[Int(ii_c122)+1] = Float32(temp_c123)
      ((ii_c122 += Int32(1))-Int32(1))
     end
     local gdis_c125::Float32 = Float32(Float64(0.0))
     local fl_c126 = Offset(d_res,Int32(Int32((Int32((Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(4)))+Int32((Int32((Int32((Int32((Int32(2)*win))+Int32(1)))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(2)))))))
     local maxdif_c127::Float32 = Float32(Int32(0))
     local k_c128::Int32 = Int32(Int32(0))
     while (k_c128<n_B)
      if (Float32((vector_in_c9[Int(k_c128)+1]-rst_c116[Int(k_c128)+1]))>=Int32(0))
       fl_c126[Int(k_c128)+1] = Float32(Float32((vector_in_c9[Int(k_c128)+1]-rst_c116[Int(k_c128)+1])))
      else
       fl_c126[Int(k_c128)+1] = Float32(Float32((rst_c116[Int(k_c128)+1]-vector_in_c9[Int(k_c128)+1])))
      end
      if (k_c128==Int32(0))
       maxdif_c127 = Float32(fl_c126[Int(k_c128)+1])
      else
       if (maxdif_c127<fl_c126[Int(k_c128)+1])
        maxdif_c127 = Float32(fl_c126[Int(k_c128)+1])
       end
      end
      ((k_c128 += Int32(1))-Int32(1))
     end
     local k_c129::Int32 = Int32(Int32(0))
     while (k_c129<n_B)
      if (vector_in_c9[Int(k_c129)+1]>=rst_c116[Int(k_c129)+1])
       fl_c126[Int(k_c129)+1] = Float32(Float32(1.))
       gdis_c125 = Float32(muladd(Float32(fl_c126[Int(k_c129)+1]),Float32(Float32((vector_in_c9[Int(k_c129)+1]-rst_c116[Int(k_c129)+1]))),Float32(gdis_c125)))
      else
       fl_c126[Int(k_c129)+1] = Float32(Float32((Float32(1.)-Float32((Float32((rst_c116[Int(k_c129)+1]-vector_in_c9[Int(k_c129)+1]))/maxdif_c127)))))
       gdis_c125 = Float32(muladd(Float32(fl_c126[Int(k_c129)+1]),Float32(Float32((rst_c116[Int(k_c129)+1]-vector_in_c9[Int(k_c129)+1]))),Float32(gdis_c125)))
      end
      ((k_c129 += Int32(1))-Int32(1))
     end
     local ormax_c130::Float32 = Float32(gdis_c125)
     local loop_times_c131::Int32 = Int32(Int32(1))
     while ((gdis_c125<=ormax_c130)&&(loop_times_c131<Int32(15)))
      loop_times_c131 = Int32(Int32((loop_times_c131+Int32(1))))
      local k_c132::Int32 = Int32(Int32(0))
      while (k_c132<n_B)
       vector_out[Int(Int32((Int32((Int32((i_c1+Int32((Int32((j_c2+StartY))*n_X))))+Int32((Int32((k_c132*n_X))*TotalY))))+Int32((Int32((Int32((y_c8*n_X))*TotalY))*n_B)))))+1] = Float32(rst_c116[Int(k_c132)+1])
       ((k_c132 += Int32(1))-Int32(1))
      end
      local savgolFilter_c133 = MVector{9,Float64}((Float64((-Float64(0.00543880))),Float64(0.0435097),Float64((-Float64(0.152289))),Float64(0.304585),Float64(0.619267),Float64(0.304585),Float64((-Float64(0.152289))),Float64(0.0435097),Float64((-Float64(0.00543880))),))
      local savgolFilterW_c134::Int32 = Int32(Int32(div(Int32(72),Int32(8))))
      local ra4W_c135::Int32 = Int32(n_B)
      local new_ra4_c136 = Offset(d_res,Int32(Int32((Int32((Int32((Int32((Int32((Int32((i_c1+Int32((j_c2*n_X))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(4)))+Int32((Int32((Int32((Int32((Int32(2)*win))+Int32(1)))*Int32((Int32((Int32(2)*win))+Int32(1)))))*Int32(3)))))+n_B))))
      local ii_c137::Int32 = Int32(Int32(0))
      while (ii_c137<Int32((Int32((savgolFilterW_c134+ra4W_c135))-Int32(1))))
       if (ii_c137<Int32(div(Int32((savgolFilterW_c134-Int32(1))),Int32(2))))
        new_ra4_c136[Int(ii_c137)+1] = Float32(Float32(((vector_in_c9[Int(Int32(0))+1]>=rst_c116[Int(Int32(0))+1]) ? vector_in_c9[Int(Int32(0))+1] : rst_c116[Int(Int32(0))+1])))
       else
        if (ii_c137>Int32((Int32((Int32(div(Int32((savgolFilterW_c134-Int32(1))),Int32(2)))+ra4W_c135))-Int32(1))))
         new_ra4_c136[Int(ii_c137)+1] = Float32(Float32(((vector_in_c9[Int(Int32((ra4W_c135-Int32(1))))+1]>=rst_c116[Int(Int32((ra4W_c135-Int32(1))))+1]) ? vector_in_c9[Int(Int32((ra4W_c135-Int32(1))))+1] : rst_c116[Int(Int32((ra4W_c135-Int32(1))))+1])))
        else
         new_ra4_c136[Int(ii_c137)+1] = Float32(Float32(((vector_in_c9[Int(Int32((ii_c137-Int32(div(Int32((savgolFilterW_c134-Int32(1))),Int32(2))))))+1]>=rst_c116[Int(Int32((ii_c137-Int32(div(Int32((savgolFilterW_c134-Int32(1))),Int32(2))))))+1]) ? vector_in_c9[Int(Int32((ii_c137-Int32(div(Int32((savgolFilterW_c134-Int32(1))),Int32(2))))))+1] : rst_c116[Int(Int32((ii_c137-Int32(div(Int32((savgolFilterW_c134-Int32(1))),Int32(2))))))+1])))
        end
       end
       ((ii_c137 += Int32(1))-Int32(1))
      end
      local ii_c138::Int32 = Int32(Int32(0))
      while (ii_c138<ra4W_c135)
       local temp_c139::Float32 = Float32(Int32(0))
       local jj_c140::Int32 = Int32(Int32(0))
       while (jj_c140<savgolFilterW_c134)
        temp_c139 = Float32(muladd(Float64(savgolFilter_c133[Int(jj_c140)+1]),Float64(new_ra4_c136[Int(Int32((ii_c138+jj_c140)))+1]),Float64(temp_c139)))
        ((jj_c140 += Int32(1))-Int32(1))
       end
       rst_c116[Int(ii_c138)+1] = Float32(temp_c139)
       ((ii_c138 += Int32(1))-Int32(1))
      end
      ormax_c130 = Float32(gdis_c125)
      gdis_c125 = Float32(Float32(0.))
      local k_c141::Int32 = Int32(Int32(0))
      while (k_c141<n_B)
       if (Float32((vector_in_c9[Int(k_c141)+1]-rst_c116[Int(k_c141)+1]))>=Int32(0))
        gdis_c125 = Float32(muladd(Float32(fl_c126[Int(k_c141)+1]),Float32(Float32((vector_in_c9[Int(k_c141)+1]-rst_c116[Int(k_c141)+1]))),Float32(gdis_c125)))
       else
        gdis_c125 = Float32(muladd(Float32(fl_c126[Int(k_c141)+1]),Float32(Float32((rst_c116[Int(k_c141)+1]-vector_in_c9[Int(k_c141)+1]))),Float32(gdis_c125)))
       end
       ((k_c141 += Int32(1))-Int32(1))
      end
     end
    end
   end
   ((y_c8 += Int32(1))-Int32(1))
  end
 end
 nothing
end
