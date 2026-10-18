@inline function comm_empty!(sigma_in,sigma_out,hamiltonian,dim::Int32,BX::Int32,BY::Int32,TX::Int32,TY::Int32,BDX::Int32)
 @inbounds begin

 end
 return
end

@inline function comm_init!(sigma_in,sigma_out,hamiltonian,dim::Int32,BX::Int32,BY::Int32,TX::Int32,TY::Int32,BDX::Int32)
 @inbounds begin
 gid = BX * BDX + TX
 sigma_id = gid * dim * dim
 for i in Int32(0):Int32(dim)-Int32(1)
  for j in Int32(0):Int32(dim)-Int32(1)
   tmp_x = 0f0
   tmp_y = 0f0
   for k in Int32(0):Int32(dim)-Int32(1)
    tmp_x += (hamiltonian[2*(i * dim + k)+1] * sigma_in[2*(sigma_id + k * dim + j)+1] -
            sigma_in[2*(sigma_id + i * dim + k)+1] * hamiltonian[2*(k * dim + j)+1])
    tmp_x -= (hamiltonian[2*(i * dim + k)+2] * sigma_in[2*(sigma_id + k * dim + j)+2] -
            sigma_in[2*(sigma_id + i * dim + k)+2] * hamiltonian[2*(k * dim + j)+2])
    tmp_y += (hamiltonian[2*(i * dim + k)+1] * sigma_in[2*(sigma_id + k * dim + j)+2] -
            sigma_in[2*(sigma_id + i * dim + k)+1] * hamiltonian[2*(k * dim + j)+2])
    tmp_y += (hamiltonian[2*(i * dim + k)+2] * sigma_in[2*(sigma_id + k * dim + j)+1] -
            sigma_in[2*(sigma_id + i * dim + k)+2] * hamiltonian[2*(k * dim + j)+1])
   end
   sigma_out[2*(sigma_id + i * dim + j)+1] += HDT * tmp_y
   sigma_out[2*(sigma_id + i * dim + j)+2] -= HDT * tmp_x
  end
 end
 end
 return
end

@inline function comm_refactor!(sigma_in,sigma_out,hamiltonian,dim::Int32,BX::Int32,BY::Int32,TX::Int32,TY::Int32,BDX::Int32)
 @inbounds begin
 gid = BX * BDX + TX
 sigma_id = gid * dim * dim * 2
 for i in Int32(0):Int32(dim)-Int32(1)
  for j in Int32(0):Int32(dim)-Int32(1)
   tmp_real = 0f0
   tmp_imag = 0f0
   for k in Int32(0):Int32(dim)-Int32(1)
    tmp_real += hamiltonian[((2 * ((i) * dim + (k))))+1] * sigma_in[((sigma_id + 2 * ((k) * dim + (j))))+1]
    tmp_real -= sigma_in[((sigma_id + 2 * ((i) * dim + (k))))+1] * hamiltonian[((2 * ((k) * dim + (j))))+1]
    tmp_real -= hamiltonian[((2 * ((i) * dim + (k)) + 1))+1] * sigma_in[((sigma_id + 2 * ((k) * dim + (j)) + 1))+1]
    tmp_real += sigma_in[((sigma_id + 2 * ((i) * dim + (k)) + 1))+1] * hamiltonian[((2 * ((k) * dim + (k)) + 1))+1]
    tmp_imag += hamiltonian[((2 * ((i) * dim + (k))))+1] * sigma_in[((sigma_id + 2 * ((k) * dim + (j)) + 1))+1]
    tmp_imag -= sigma_in[((sigma_id + 2 * ((i) * dim + (k))))+1] * hamiltonian[((2 * ((k) * dim + (k)) + 1))+1]
    tmp_imag += hamiltonian[((2 * ((i) * dim + (k)) + 1))+1] * sigma_in[((sigma_id + 2 * ((k) * dim + (j))))+1]
    tmp_imag -= sigma_in[((sigma_id + 2 * ((i) * dim + (k)) + 1))+1] * hamiltonian[((2 * ((k) * dim + (j))))+1]
   end
   sigma_out[((sigma_id + 2 * ((i) * dim + (j))))+1] += HDT * tmp_imag
   sigma_out[((sigma_id + 2 * ((i) * dim + (j)) + 1))+1] -= HDT * tmp_real
  end
 end
 end
 return
end

@inline function comm_refactor_direct_store!(sigma_in,sigma_out,hamiltonian,dim::Int32,BX::Int32,BY::Int32,TX::Int32,TY::Int32,BDX::Int32)
 @inbounds begin
 gid = BX * BDX + TX
 sigma_id = gid * dim * dim * 2
 for i in Int32(0):Int32(dim)-Int32(1)
  for j in Int32(0):Int32(dim)-Int32(1)
   for k in Int32(0):Int32(dim)-Int32(1)
    sigma_out[((sigma_id + 2 * ((i) * dim + (j))))+1] += hamiltonian[((2 * ((i) * dim + (k))))+1] * sigma_in[((sigma_id + 2 * ((k) * dim + (j)) + 1))+1]
    sigma_out[((sigma_id + 2 * ((i) * dim + (j))))+1] -= sigma_in[((sigma_id + 2 * ((i) * dim + (k))))+1] * hamiltonian[((2 * ((k) * dim + (k)) + 1))+1]
    sigma_out[((sigma_id + 2 * ((i) * dim + (j))))+1] += hamiltonian[((2 * ((i) * dim + (k)) + 1))+1] * sigma_in[((sigma_id + 2 * ((k) * dim + (j))))+1]
    sigma_out[((sigma_id + 2 * ((i) * dim + (j))))+1] -= sigma_in[((sigma_id + 2 * ((i) * dim + (k)) + 1))+1] * hamiltonian[((2 * ((k) * dim + (j))))+1]
    sigma_out[((sigma_id + 2 * ((i) * dim + (j)) + 1))+1] -= hamiltonian[((2 * ((i) * dim + (k))))+1] * sigma_in[((sigma_id + 2 * ((k) * dim + (j))))+1]
    sigma_out[((sigma_id + 2 * ((i) * dim + (j)) + 1))+1] += sigma_in[((sigma_id + 2 * ((i) * dim + (k))))+1] * hamiltonian[((2 * ((k) * dim + (j))))+1]
    sigma_out[((sigma_id + 2 * ((i) * dim + (j)) + 1))+1] += hamiltonian[((2 * ((i) * dim + (k)) + 1))+1] * sigma_in[((sigma_id + 2 * ((k) * dim + (j)) + 1))+1]
    sigma_out[((sigma_id + 2 * ((i) * dim + (j)) + 1))+1] -= sigma_in[((sigma_id + 2 * ((i) * dim + (k)) + 1))+1] * hamiltonian[((2 * ((k) * dim + (k)) + 1))+1]
   end
  end
 end
 end
 return
end

@inline function comm_aosoa_naive!(sigma_in,sigma_out,hamiltonian,dim::Int32,BX::Int32,BY::Int32,TX::Int32,TY::Int32,BDX::Int32)
 @inbounds begin
 gid = BX * BDX + TX
 for i in Int32(0):Int32(dim)-Int32(1)
  for j in Int32(0):Int32(dim)-Int32(1)
   tmp_real = 0f0
   tmp_imag = 0f0
   for k in Int32(0):Int32(dim)-Int32(1)
    tmp_imag -= hamiltonian[(((i) * dim + (k)))+1] * sigma_in[((((gid ÷ 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (k) + (j)) + ((gid % 4))))+1]
    tmp_imag += sigma_in[((((gid ÷ 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (i) + (k)) + ((gid % 4))))+1] * hamiltonian[(((k) * dim + (j)))+1]
    tmp_imag += hamiltonian[((dim * dim + (i) * dim + (k)))+1] * sigma_in[((((gid ÷ 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (k) + (j)) + 4 + ((gid % 4))))+1]
    tmp_imag -= sigma_in[((((gid ÷ 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (i) + (k)) + 4 + ((gid % 4))))+1] * hamiltonian[((dim * dim + (k) * dim + (j)))+1]
    tmp_real += hamiltonian[(((i) * dim + (k)))+1] * sigma_in[((((gid ÷ 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (k) + (j)) + 4 + ((gid % 4))))+1]
    tmp_real -= sigma_in[((((gid ÷ 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (i) + (k)) + ((gid % 4))))+1] * hamiltonian[((dim * dim + (k) * dim + (j)))+1]
    tmp_real += hamiltonian[((dim * dim + (i) * dim + (k)))+1] * sigma_in[((((gid ÷ 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (k) + (j)) + ((gid % 4))))+1]
    tmp_real -= sigma_in[((((gid ÷ 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (i) + (k)) + 4 + ((gid % 4))))+1] * hamiltonian[(((k) * dim + (j)))+1]
   end
   sigma_out[((((gid ÷ 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (i) + (j)) + ((gid % 4))))+1] += tmp_real
   sigma_out[((((gid ÷ 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (i) + (j)) + 4 + ((gid % 4))))+1] += tmp_imag
  end
 end
 end
 return
end

@inline function comm_aosoa_naive_constants!(sigma_in,sigma_out,hamiltonian,dim::Int32,BX::Int32,BY::Int32,TX::Int32,TY::Int32,BDX::Int32)
 @inbounds begin
 gid = BX * BDX + TX
 for i in Int32(0):Int32(7)-Int32(1)
  for j in Int32(0):Int32(7)-Int32(1)
   tmp_real = 0f0
   tmp_imag = 0f0
   for k in Int32(0):Int32(7)-Int32(1)
    tmp_imag -= hamiltonian[(((i) * 7 + (k)))+1] * sigma_in[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (k) + (j)) + ((gid % 4))))+1]
    tmp_imag += sigma_in[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (k)) + ((gid % 4))))+1] * hamiltonian[(((k) * 7 + (j)))+1]
    tmp_imag += hamiltonian[((7 * 7 + (i) * 7 + (k)))+1] * sigma_in[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (k) + (j)) + 4 + ((gid % 4))))+1]
    tmp_imag -= sigma_in[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (k)) + 4 + ((gid % 4))))+1] * hamiltonian[((7 * 7 + (k) * 7 + (j)))+1]
    tmp_real += hamiltonian[(((i) * 7 + (k)))+1] * sigma_in[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (k) + (j)) + 4 + ((gid % 4))))+1]
    tmp_real -= sigma_in[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (k)) + ((gid % 4))))+1] * hamiltonian[((7 * 7 + (k) * 7 + (j)))+1]
    tmp_real += hamiltonian[((7 * 7 + (i) * 7 + (k)))+1] * sigma_in[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (k) + (j)) + ((gid % 4))))+1]
    tmp_real -= sigma_in[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (k)) + 4 + ((gid % 4))))+1] * hamiltonian[(((k) * 7 + (j)))+1]
   end
   sigma_out[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + ((gid % 4))))+1] += tmp_real
   sigma_out[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + 4 + ((gid % 4))))+1] += tmp_imag
  end
 end
 end
 return
end

@inline function comm_aosoa_naive_constants_perm!(sigma_in,sigma_out,hamiltonian,dim::Int32,BX::Int32,BY::Int32,TX::Int32,TY::Int32,BDX::Int32)
 @inbounds begin
 gid = BX * BDX + TX
 for i in Int32(0):Int32(7)-Int32(1)
  for k in Int32(0):Int32(7)-Int32(1)
   ham_real_tmp = hamiltonian[(((i) * 7 + (k)))+1]
   ham_imag_tmp = hamiltonian[((7 * 7 + (i) * 7 + (k)))+1]
   sigma_real_tmp = sigma_in[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (k)) + ((gid % 4))))+1]
   sigma_imag_tmp = sigma_in[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (k)) + 4 + ((gid % 4))))+1]
   for j in Int32(0):Int32(7)-Int32(1)
    tmp_real = sigma_out[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + ((gid % 4))))+1]
    tmp_imag = sigma_out[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + 4 + ((gid % 4))))+1]
    tmp_imag -= ham_real_tmp * sigma_in[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (k) + (j)) + ((gid % 4))))+1]
    tmp_imag += sigma_real_tmp * hamiltonian[(((k) * 7 + (j)))+1]
    tmp_imag += ham_imag_tmp * sigma_in[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (k) + (j)) + 4 + ((gid % 4))))+1]
    tmp_imag -= sigma_imag_tmp * hamiltonian[((7 * 7 + (k) * 7 + (j)))+1]
    tmp_real += ham_real_tmp * sigma_in[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (k) + (j)) + 4 + ((gid % 4))))+1]
    tmp_real -= sigma_real_tmp * hamiltonian[((7 * 7 + (k) * 7 + (j)))+1]
    tmp_real += ham_imag_tmp * sigma_in[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (k) + (j)) + ((gid % 4))))+1]
    tmp_real -= sigma_imag_tmp * hamiltonian[(((k) * 7 + (j)))+1]
    sigma_out[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + ((gid % 4))))+1] = tmp_real
    sigma_out[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + 4 + ((gid % 4))))+1] = tmp_imag
   end
  end
 end
 end
 return
end

@inline function comm_aosoa_naive_direct!(sigma_in,sigma_out,hamiltonian,dim::Int32,BX::Int32,BY::Int32,TX::Int32,TY::Int32,BDX::Int32)
 @inbounds begin
 gid = BX * BDX + TX
 for i in Int32(0):Int32(dim)-Int32(1)
  for j in Int32(0):Int32(dim)-Int32(1)
   for k in Int32(0):Int32(dim)-Int32(1)
    sigma_out[((((gid ÷ 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (i) + (j)) + 4 + ((gid % 4))))+1] -= hamiltonian[(((i) * dim + (k)))+1] * sigma_in[((((gid ÷ 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (k) + (j)) + ((gid % 4))))+1]
    sigma_out[((((gid ÷ 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (i) + (j)) + 4 + ((gid % 4))))+1] += sigma_in[((((gid ÷ 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (i) + (k)) + ((gid % 4))))+1] * hamiltonian[(((k) * dim + (j)))+1]
    sigma_out[((((gid ÷ 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (i) + (j)) + 4 + ((gid % 4))))+1] += hamiltonian[((dim * dim + (i) * dim + (k)))+1] * sigma_in[((((gid ÷ 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (k) + (j)) + 4 + ((gid % 4))))+1]
    sigma_out[((((gid ÷ 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (i) + (j)) + 4 + ((gid % 4))))+1] -= sigma_in[((((gid ÷ 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (i) + (k)) + 4 + ((gid % 4))))+1] * hamiltonian[((dim * dim + (k) * dim + (j)))+1]
    sigma_out[((((gid ÷ 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (i) + (j)) + ((gid % 4))))+1] += hamiltonian[(((i) * dim + (k)))+1] * sigma_in[((((gid ÷ 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (k) + (j)) + 4 + ((gid % 4))))+1]
    sigma_out[((((gid ÷ 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (i) + (j)) + ((gid % 4))))+1] -= sigma_in[((((gid ÷ 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (i) + (k)) + ((gid % 4))))+1] * hamiltonian[((dim * dim + (k) * dim + (j)))+1]
    sigma_out[((((gid ÷ 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (i) + (j)) + ((gid % 4))))+1] += hamiltonian[((dim * dim + (i) * dim + (k)))+1] * sigma_in[((((gid ÷ 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (k) + (j)) + ((gid % 4))))+1]
    sigma_out[((((gid ÷ 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (i) + (j)) + ((gid % 4))))+1] -= sigma_in[((((gid ÷ 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (i) + (k)) + 4 + ((gid % 4))))+1] * hamiltonian[(((k) * dim + (j)))+1]
   end
  end
 end
 end
 return
end

@inline function comm_aosoa_naive_constants_direct!(sigma_in,sigma_out,hamiltonian,dim::Int32,BX::Int32,BY::Int32,TX::Int32,TY::Int32,BDX::Int32)
 @inbounds begin
 gid = BX * BDX + TX
 for i in Int32(0):Int32(7)-Int32(1)
  for j in Int32(0):Int32(7)-Int32(1)
   for k in Int32(0):Int32(7)-Int32(1)
    sigma_out[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + 4 + ((gid % 4))))+1] -= hamiltonian[(((i) * 7 + (k)))+1] * sigma_in[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (k) + (j)) + ((gid % 4))))+1]
    sigma_out[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + 4 + ((gid % 4))))+1] += sigma_in[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (k)) + ((gid % 4))))+1] * hamiltonian[(((k) * 7 + (j)))+1]
    sigma_out[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + 4 + ((gid % 4))))+1] += hamiltonian[((7 * 7 + (i) * 7 + (k)))+1] * sigma_in[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (k) + (j)) + 4 + ((gid % 4))))+1]
    sigma_out[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + 4 + ((gid % 4))))+1] -= sigma_in[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (k)) + 4 + ((gid % 4))))+1] * hamiltonian[((7 * 7 + (k) * 7 + (j)))+1]
    sigma_out[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + ((gid % 4))))+1] += hamiltonian[(((i) * 7 + (k)))+1] * sigma_in[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (k) + (j)) + 4 + ((gid % 4))))+1]
    sigma_out[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + ((gid % 4))))+1] -= sigma_in[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (k)) + ((gid % 4))))+1] * hamiltonian[((7 * 7 + (k) * 7 + (j)))+1]
    sigma_out[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + ((gid % 4))))+1] += hamiltonian[((7 * 7 + (i) * 7 + (k)))+1] * sigma_in[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (k) + (j)) + ((gid % 4))))+1]
    sigma_out[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + ((gid % 4))))+1] -= sigma_in[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (k)) + 4 + ((gid % 4))))+1] * hamiltonian[(((k) * 7 + (j)))+1]
   end
  end
 end
 end
 return
end

@inline function comm_aosoa_naive_constants_direct_perm!(sigma_in,sigma_out,hamiltonian,dim::Int32,BX::Int32,BY::Int32,TX::Int32,TY::Int32,BDX::Int32)
 @inbounds begin
 gid = BX * BDX + TX
 for i in Int32(0):Int32(7)-Int32(1)
  for k in Int32(0):Int32(7)-Int32(1)
   ham_real_tmp = hamiltonian[(((i) * 7 + (k)))+1]
   ham_imag_tmp = hamiltonian[((7 * 7 + (i) * 7 + (k)))+1]
   sigma_real_tmp = sigma_in[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (k)) + ((gid % 4))))+1]
   sigma_imag_tmp = sigma_in[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (k)) + 4 + ((gid % 4))))+1]
   for j in Int32(0):Int32(7)-Int32(1)
    sigma_out[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + 4 + ((gid % 4))))+1] -= ham_real_tmp * sigma_in[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (k) + (j)) + ((gid % 4))))+1]
    sigma_out[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + 4 + ((gid % 4))))+1] += sigma_real_tmp * hamiltonian[(((k) * 7 + (j)))+1]
    sigma_out[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + 4 + ((gid % 4))))+1] += ham_imag_tmp * sigma_in[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (k) + (j)) + 4 + ((gid % 4))))+1]
    sigma_out[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + 4 + ((gid % 4))))+1] -= sigma_imag_tmp * hamiltonian[((7 * 7 + (k) * 7 + (j)))+1]
    sigma_out[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + ((gid % 4))))+1] += ham_real_tmp * sigma_in[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (k) + (j)) + 4 + ((gid % 4))))+1]
    sigma_out[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + ((gid % 4))))+1] -= sigma_real_tmp * hamiltonian[((7 * 7 + (k) * 7 + (j)))+1]
    sigma_out[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + ((gid % 4))))+1] += ham_imag_tmp * sigma_in[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (k) + (j)) + ((gid % 4))))+1]
    sigma_out[((((gid ÷ 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + ((gid % 4))))+1] -= sigma_imag_tmp * hamiltonian[(((k) * 7 + (j)))+1]
   end
  end
 end
 end
 return
end

@inline function comm_aosoa!(sigma_in,sigma_out,hamiltonian,dim::Int32,BX::Int32,BY::Int32,TX::Int32,TY::Int32,BDX::Int32)
 @inbounds begin
 for i in Int32(0):Int32(dim)-Int32(1)
  for j in Int32(0):Int32(dim)-Int32(1)
   tmp_real = 0f0
   tmp_imag = 0f0
   for k in Int32(0):Int32(dim)-Int32(1)
    tmp_imag -= hamiltonian[(((i) * dim + (k)))+1] * sigma_in[((((64 * BY + TY) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (k) + (j)) + TX))+1]
    tmp_imag += sigma_in[((((64 * BY + TY) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (i) + (k)) + TX))+1] * hamiltonian[(((k) * dim + (j)))+1]
    tmp_imag += hamiltonian[((dim * dim + (i) * dim + (k)))+1] * sigma_in[((((64 * BY + TY) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (k) + (j)) + 4 + TX))+1]
    tmp_imag -= sigma_in[((((64 * BY + TY) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (i) + (k)) + 4 + TX))+1] * hamiltonian[((dim * dim + (k) * dim + (j)))+1]
    tmp_real += hamiltonian[(((i) * dim + (k)))+1] * sigma_in[((((64 * BY + TY) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (k) + (j)) + 4 + TX))+1]
    tmp_real -= sigma_in[((((64 * BY + TY) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (i) + (k)) + TX))+1] * hamiltonian[((dim * dim + (k) * dim + (j)))+1]
    tmp_real += hamiltonian[((dim * dim + (i) * dim + (k)))+1] * sigma_in[((((64 * BY + TY) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (k) + (j)) + TX))+1]
    tmp_real -= sigma_in[((((64 * BY + TY) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (i) + (k)) + 4 + TX))+1] * hamiltonian[(((k) * dim + (j)))+1]
   end
   sigma_out[((((64 * BY + TY) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (i) + (j)) + TX))+1] += tmp_real
   sigma_out[((((64 * BY + TY) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (i) + (j)) + 4 + TX))+1] += tmp_imag
  end
 end
 end
 return
end

@inline function comm_aosoa_constants!(sigma_in,sigma_out,hamiltonian,dim::Int32,BX::Int32,BY::Int32,TX::Int32,TY::Int32,BDX::Int32)
 @inbounds begin
 for i in Int32(0):Int32(7)-Int32(1)
  for j in Int32(0):Int32(7)-Int32(1)
   tmp_real = 0f0
   tmp_imag = 0f0
   for k in Int32(0):Int32(7)-Int32(1)
    tmp_imag -= hamiltonian[(((i) * 7 + (k)))+1] * sigma_in[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (k) + (j)) + TX))+1]
    tmp_imag += sigma_in[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (k)) + TX))+1] * hamiltonian[(((k) * 7 + (j)))+1]
    tmp_imag += hamiltonian[((7 * 7 + (i) * 7 + (k)))+1] * sigma_in[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (k) + (j)) + 4 + TX))+1]
    tmp_imag -= sigma_in[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (k)) + 4 + TX))+1] * hamiltonian[((7 * 7 + (k) * 7 + (j)))+1]
    tmp_real += hamiltonian[(((i) * 7 + (k)))+1] * sigma_in[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (k) + (j)) + 4 + TX))+1]
    tmp_real -= sigma_in[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (k)) + TX))+1] * hamiltonian[((7 * 7 + (k) * 7 + (j)))+1]
    tmp_real += hamiltonian[((7 * 7 + (i) * 7 + (k)))+1] * sigma_in[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (k) + (j)) + TX))+1]
    tmp_real -= sigma_in[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (k)) + 4 + TX))+1] * hamiltonian[(((k) * 7 + (j)))+1]
   end
   sigma_out[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + TX))+1] += tmp_real
   sigma_out[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + 4 + TX))+1] += tmp_imag
  end
 end
 end
 return
end

@inline function comm_aosoa_constants_perm!(sigma_in,sigma_out,hamiltonian,dim::Int32,BX::Int32,BY::Int32,TX::Int32,TY::Int32,BDX::Int32)
 @inbounds begin
 for i in Int32(0):Int32(7)-Int32(1)
  for k in Int32(0):Int32(7)-Int32(1)
   ham_real_tmp = hamiltonian[(((i) * 7 + (k)))+1]
   ham_imag_tmp = hamiltonian[((7 * 7 + (i) * 7 + (k)))+1]
   sigma_real_tmp = sigma_in[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (k)) + TX))+1]
   sigma_imag_tmp = sigma_in[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (k)) + 4 + TX))+1]
   for j in Int32(0):Int32(7)-Int32(1)
    tmp_real = sigma_out[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + TX))+1]
    tmp_imag = sigma_out[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + 4 + TX))+1]
    tmp_imag -= ham_real_tmp * sigma_in[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (k) + (j)) + TX))+1]
    tmp_imag += sigma_real_tmp * hamiltonian[(((k) * 7 + (j)))+1]
    tmp_imag += ham_imag_tmp * sigma_in[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (k) + (j)) + 4 + TX))+1]
    tmp_imag -= sigma_imag_tmp * hamiltonian[((7 * 7 + (k) * 7 + (j)))+1]
    tmp_real += ham_real_tmp * sigma_in[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (k) + (j)) + 4 + TX))+1]
    tmp_real -= sigma_real_tmp * hamiltonian[((7 * 7 + (k) * 7 + (j)))+1]
    tmp_real += ham_imag_tmp * sigma_in[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (k) + (j)) + TX))+1]
    tmp_real -= sigma_imag_tmp * hamiltonian[(((k) * 7 + (j)))+1]
    sigma_out[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + TX))+1] = tmp_real
    sigma_out[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + 4 + TX))+1] = tmp_imag
   end
  end
 end
 end
 return
end

@inline function comm_aosoa_direct!(sigma_in,sigma_out,hamiltonian,dim::Int32,BX::Int32,BY::Int32,TX::Int32,TY::Int32,BDX::Int32)
 @inbounds begin
 for i in Int32(0):Int32(dim)-Int32(1)
  for j in Int32(0):Int32(dim)-Int32(1)
   for k in Int32(0):Int32(dim)-Int32(1)
    sigma_out[((((64 * BY + TY) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (i) + (j)) + 4 + TX))+1] -= hamiltonian[(((i) * dim + (k)))+1] * sigma_in[((((64 * BY + TY) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (k) + (j)) + TX))+1]
    sigma_out[((((64 * BY + TY) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (i) + (j)) + 4 + TX))+1] += sigma_in[((((64 * BY + TY) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (i) + (k)) + TX))+1] * hamiltonian[(((k) * dim + (j)))+1]
    sigma_out[((((64 * BY + TY) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (i) + (j)) + 4 + TX))+1] += hamiltonian[((dim * dim + (i) * dim + (k)))+1] * sigma_in[((((64 * BY + TY) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (k) + (j)) + 4 + TX))+1]
    sigma_out[((((64 * BY + TY) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (i) + (j)) + 4 + TX))+1] -= sigma_in[((((64 * BY + TY) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (i) + (k)) + 4 + TX))+1] * hamiltonian[((dim * dim + (k) * dim + (j)))+1]
    sigma_out[((((64 * BY + TY) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (i) + (j)) + TX))+1] += hamiltonian[(((i) * dim + (k)))+1] * sigma_in[((((64 * BY + TY) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (k) + (j)) + 4 + TX))+1]
    sigma_out[((((64 * BY + TY) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (i) + (j)) + TX))+1] -= sigma_in[((((64 * BY + TY) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (i) + (k)) + TX))+1] * hamiltonian[((dim * dim + (k) * dim + (j)))+1]
    sigma_out[((((64 * BY + TY) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (i) + (j)) + TX))+1] += hamiltonian[((dim * dim + (i) * dim + (k)))+1] * sigma_in[((((64 * BY + TY) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (k) + (j)) + TX))+1]
    sigma_out[((((64 * BY + TY) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (i) + (j)) + TX))+1] -= sigma_in[((((64 * BY + TY) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (i) + (k)) + 4 + TX))+1] * hamiltonian[(((k) * dim + (j)))+1]
   end
  end
 end
 end
 return
end

@inline function comm_aosoa_constants_direct!(sigma_in,sigma_out,hamiltonian,dim::Int32,BX::Int32,BY::Int32,TX::Int32,TY::Int32,BDX::Int32)
 @inbounds begin
 for i in Int32(0):Int32(7)-Int32(1)
  for j in Int32(0):Int32(7)-Int32(1)
   for k in Int32(0):Int32(7)-Int32(1)
    sigma_out[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + 4 + TX))+1] -= hamiltonian[(((i) * 7 + (k)))+1] * sigma_in[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (k) + (j)) + TX))+1]
    sigma_out[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + 4 + TX))+1] += sigma_in[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (k)) + TX))+1] * hamiltonian[(((k) * 7 + (j)))+1]
    sigma_out[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + 4 + TX))+1] += hamiltonian[((7 * 7 + (i) * 7 + (k)))+1] * sigma_in[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (k) + (j)) + 4 + TX))+1]
    sigma_out[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + 4 + TX))+1] -= sigma_in[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (k)) + 4 + TX))+1] * hamiltonian[((7 * 7 + (k) * 7 + (j)))+1]
    sigma_out[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + TX))+1] += hamiltonian[(((i) * 7 + (k)))+1] * sigma_in[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (k) + (j)) + 4 + TX))+1]
    sigma_out[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + TX))+1] -= sigma_in[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (k)) + TX))+1] * hamiltonian[((7 * 7 + (k) * 7 + (j)))+1]
    sigma_out[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + TX))+1] += hamiltonian[((7 * 7 + (i) * 7 + (k)))+1] * sigma_in[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (k) + (j)) + TX))+1]
    sigma_out[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + TX))+1] -= sigma_in[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (k)) + 4 + TX))+1] * hamiltonian[(((k) * 7 + (j)))+1]
   end
  end
 end
 end
 return
end

@inline function comm_aosoa_constants_direct_perm!(sigma_in,sigma_out,hamiltonian,dim::Int32,BX::Int32,BY::Int32,TX::Int32,TY::Int32,BDX::Int32)
 @inbounds begin
 for i in Int32(0):Int32(7)-Int32(1)
  for k in Int32(0):Int32(7)-Int32(1)
   ham_real_tmp = hamiltonian[(((i) * 7 + (k)))+1]
   ham_imag_tmp = hamiltonian[((7 * 7 + (i) * 7 + (k)))+1]
   sigma_real_tmp = sigma_in[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (k)) + TX))+1]
   sigma_imag_tmp = sigma_in[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (k)) + 4 + TX))+1]
   for j in Int32(0):Int32(7)-Int32(1)
    sigma_out[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + 4 + TX))+1] -= ham_real_tmp * sigma_in[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (k) + (j)) + TX))+1]
    sigma_out[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + 4 + TX))+1] += sigma_real_tmp * hamiltonian[(((k) * 7 + (j)))+1]
    sigma_out[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + 4 + TX))+1] += ham_imag_tmp * sigma_in[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (k) + (j)) + 4 + TX))+1]
    sigma_out[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + 4 + TX))+1] -= sigma_imag_tmp * hamiltonian[((7 * 7 + (k) * 7 + (j)))+1]
    sigma_out[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + TX))+1] += ham_real_tmp * sigma_in[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (k) + (j)) + 4 + TX))+1]
    sigma_out[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + TX))+1] -= sigma_real_tmp * hamiltonian[((7 * 7 + (k) * 7 + (j)))+1]
    sigma_out[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + TX))+1] += ham_imag_tmp * sigma_in[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (k) + (j)) + TX))+1]
    sigma_out[((((64 * BY + TY) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + TX))+1] -= sigma_imag_tmp * hamiltonian[(((k) * 7 + (j)))+1]
   end
  end
 end
 end
 return
end

@inline function comm_manual_aosoa!(sigma_in,sigma_out,hamiltonian,dim::Int32,BX::Int32,BY::Int32,TX::Int32,TY::Int32,BDX::Int32)
 @inbounds begin
 gid = BX * BDX + TX
 for i in Int32(0):Int32(dim)-Int32(1)
  for j in Int32(0):Int32(dim)-Int32(1)
   tmp_real = StudyV4(0f0)
   tmp_imag = StudyV4(0f0)
   for k in Int32(0):Int32(dim)-Int32(1)
    tmp_imag -= hamiltonian[(((i) * dim + (k)))+1] * sigma_in[(((gid * dim * dim * 2) + 2 * (dim * (k) + (j))))+1]
    tmp_imag += sigma_in[(((gid * dim * dim * 2) + 2 * (dim * (i) + (k))))+1] * hamiltonian[(((k) * dim + (j)))+1]
    tmp_imag += hamiltonian[((dim * dim + (i) * dim + (k)))+1] * sigma_in[(((gid * dim * dim * 2) + 2 * (dim * (k) + (j)) + 1))+1]
    tmp_imag -= sigma_in[(((gid * dim * dim * 2) + 2 * (dim * (i) + (k)) + 1))+1] * hamiltonian[((dim * dim + (k) * dim + (j)))+1]
    tmp_real += hamiltonian[(((i) * dim + (k)))+1] * sigma_in[(((gid * dim * dim * 2) + 2 * (dim * (k) + (j)) + 1))+1]
    tmp_real -= sigma_in[(((gid * dim * dim * 2) + 2 * (dim * (i) + (k))))+1] * hamiltonian[((dim * dim + (k) * dim + (j)))+1]
    tmp_real += hamiltonian[((dim * dim + (i) * dim + (k)))+1] * sigma_in[(((gid * dim * dim * 2) + 2 * (dim * (k) + (j))))+1]
    tmp_real -= sigma_in[(((gid * dim * dim * 2) + 2 * (dim * (i) + (k)) + 1))+1] * hamiltonian[(((k) * dim + (j)))+1]
   end
   sigma_out[(((gid * dim * dim * 2) + 2 * (dim * (i) + (j))))+1] += tmp_real
   sigma_out[(((gid * dim * dim * 2) + 2 * (dim * (i) + (j)) + 1))+1] += tmp_imag
  end
 end
 end
 return
end

@inline function comm_manual_aosoa_constants!(sigma_in,sigma_out,hamiltonian,dim::Int32,BX::Int32,BY::Int32,TX::Int32,TY::Int32,BDX::Int32)
 @inbounds begin
 gid = BX * BDX + TX
 for i in Int32(0):Int32(7)-Int32(1)
  for j in Int32(0):Int32(7)-Int32(1)
   tmp_real = StudyV4(0f0)
   tmp_imag = StudyV4(0f0)
   for k in Int32(0):Int32(7)-Int32(1)
    tmp_imag -= hamiltonian[(((i) * 7 + (k)))+1] * sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j))))+1]
    tmp_imag += sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k))))+1] * hamiltonian[(((k) * 7 + (j)))+1]
    tmp_imag += hamiltonian[((7 * 7 + (i) * 7 + (k)))+1] * sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j)) + 1))+1]
    tmp_imag -= sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k)) + 1))+1] * hamiltonian[((7 * 7 + (k) * 7 + (j)))+1]
    tmp_real += hamiltonian[(((i) * 7 + (k)))+1] * sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j)) + 1))+1]
    tmp_real -= sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k))))+1] * hamiltonian[((7 * 7 + (k) * 7 + (j)))+1]
    tmp_real += hamiltonian[((7 * 7 + (i) * 7 + (k)))+1] * sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j))))+1]
    tmp_real -= sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k)) + 1))+1] * hamiltonian[(((k) * 7 + (j)))+1]
   end
   sigma_out[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j))))+1] += tmp_real
   sigma_out[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)) + 1))+1] += tmp_imag
  end
 end
 end
 return
end

@inline function comm_manual_aosoa_constants_perm!(sigma_in,sigma_out,hamiltonian,dim::Int32,BX::Int32,BY::Int32,TX::Int32,TY::Int32,BDX::Int32)
 @inbounds begin
 gid = BX * BDX + TX
 for i in Int32(0):Int32(7)-Int32(1)
  for k in Int32(0):Int32(7)-Int32(1)
   ham_real_tmp = StudyV4(hamiltonian[(((i) * 7 + (k)))+1])
   ham_imag_tmp = StudyV4(hamiltonian[((7 * 7 + (i) * 7 + (k)))+1])
   sigma_real_tmp = sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k))))+1]
   sigma_imag_tmp = sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k)) + 1))+1]
   for j in Int32(0):Int32(7)-Int32(1)
    tmp_real = sigma_out[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j))))+1]
    tmp_imag = sigma_out[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)) + 1))+1]
    tmp_imag -= ham_real_tmp * sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j))))+1]
    tmp_imag += sigma_real_tmp * hamiltonian[(((k) * 7 + (j)))+1]
    tmp_imag += ham_imag_tmp * sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j)) + 1))+1]
    tmp_imag -= sigma_imag_tmp * hamiltonian[((7 * 7 + (k) * 7 + (j)))+1]
    tmp_real += ham_real_tmp * sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j)) + 1))+1]
    tmp_real -= sigma_real_tmp * hamiltonian[((7 * 7 + (k) * 7 + (j)))+1]
    tmp_real += ham_imag_tmp * sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j))))+1]
    tmp_real -= sigma_imag_tmp * hamiltonian[(((k) * 7 + (j)))+1]
    sigma_out[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j))))+1] = tmp_real
    sigma_out[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)) + 1))+1] = tmp_imag
   end
  end
 end
 end
 return
end

@inline function comm_manual_aosoa_constants_perm_prefetch!(sigma_in,sigma_out,hamiltonian,dim::Int32,BX::Int32,BY::Int32,TX::Int32,TY::Int32,BDX::Int32)
 @inbounds begin
 gid = BX * BDX + TX
 for i in Int32(0):Int32(7)-Int32(1)
  j = 0
  for j in Int32(0):Int32(7)-Int32(1)
   tmp_real = StudyV4(0f0)
   tmp_imag = StudyV4(0f0)
   for k in Int32(0):Int32(7)-Int32(1)
    tmp_imag -= hamiltonian[(((i) * 7 + (k)))+1] * sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j))))+1]
    tmp_imag += sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k))))+1] * hamiltonian[(((k) * 7 + (j)))+1]
    tmp_imag += hamiltonian[((7 * 7 + (i) * 7 + (k)))+1] * sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j)) + 1))+1]
    tmp_imag -= sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k)) + 1))+1] * hamiltonian[((7 * 7 + (k) * 7 + (j)))+1]
    tmp_real += hamiltonian[(((i) * 7 + (k)))+1] * sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j)) + 1))+1]
    tmp_real -= sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k))))+1] * hamiltonian[((7 * 7 + (k) * 7 + (j)))+1]
    tmp_real += hamiltonian[((7 * 7 + (i) * 7 + (k)))+1] * sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j))))+1]
    tmp_real -= sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k)) + 1))+1] * hamiltonian[(((k) * 7 + (j)))+1]
   end
   sigma_out[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j))))+1] += tmp_real
   sigma_out[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)) + 1))+1] += tmp_imag
  end
 end
 end
 return
end

@inline function comm_manual_aosoa_direct!(sigma_in,sigma_out,hamiltonian,dim::Int32,BX::Int32,BY::Int32,TX::Int32,TY::Int32,BDX::Int32)
 @inbounds begin
 gid = BX * BDX + TX
 for i in Int32(0):Int32(dim)-Int32(1)
  for j in Int32(0):Int32(dim)-Int32(1)
   for k in Int32(0):Int32(dim)-Int32(1)
    sigma_out[(((gid * dim * dim * 2) + 2 * (dim * (i) + (j)) + 1))+1] -= hamiltonian[(((i) * dim + (k)))+1] * sigma_in[(((gid * dim * dim * 2) + 2 * (dim * (k) + (j))))+1]
    sigma_out[(((gid * dim * dim * 2) + 2 * (dim * (i) + (j)) + 1))+1] += sigma_in[(((gid * dim * dim * 2) + 2 * (dim * (i) + (k))))+1] * hamiltonian[(((k) * dim + (j)))+1]
    sigma_out[(((gid * dim * dim * 2) + 2 * (dim * (i) + (j)) + 1))+1] += hamiltonian[((dim * dim + (i) * dim + (k)))+1] * sigma_in[(((gid * dim * dim * 2) + 2 * (dim * (k) + (j)) + 1))+1]
    sigma_out[(((gid * dim * dim * 2) + 2 * (dim * (i) + (j)) + 1))+1] -= sigma_in[(((gid * dim * dim * 2) + 2 * (dim * (i) + (k)) + 1))+1] * hamiltonian[((dim * dim + (k) * dim + (j)))+1]
    sigma_out[(((gid * dim * dim * 2) + 2 * (dim * (i) + (j))))+1] += hamiltonian[(((i) * dim + (k)))+1] * sigma_in[(((gid * dim * dim * 2) + 2 * (dim * (k) + (j)) + 1))+1]
    sigma_out[(((gid * dim * dim * 2) + 2 * (dim * (i) + (j))))+1] -= sigma_in[(((gid * dim * dim * 2) + 2 * (dim * (i) + (k))))+1] * hamiltonian[((dim * dim + (k) * dim + (j)))+1]
    sigma_out[(((gid * dim * dim * 2) + 2 * (dim * (i) + (j))))+1] += hamiltonian[((dim * dim + (i) * dim + (k)))+1] * sigma_in[(((gid * dim * dim * 2) + 2 * (dim * (k) + (j))))+1]
    sigma_out[(((gid * dim * dim * 2) + 2 * (dim * (i) + (j))))+1] -= sigma_in[(((gid * dim * dim * 2) + 2 * (dim * (i) + (k)) + 1))+1] * hamiltonian[(((k) * dim + (j)))+1]
   end
  end
 end
 end
 return
end

@inline function comm_manual_aosoa_constants_direct!(sigma_in,sigma_out,hamiltonian,dim::Int32,BX::Int32,BY::Int32,TX::Int32,TY::Int32,BDX::Int32)
 @inbounds begin
 gid = BX * BDX + TX
 for i in Int32(0):Int32(7)-Int32(1)
  for j in Int32(0):Int32(7)-Int32(1)
   for k in Int32(0):Int32(7)-Int32(1)
    sigma_out[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)) + 1))+1] -= hamiltonian[(((i) * 7 + (k)))+1] * sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j))))+1]
    sigma_out[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)) + 1))+1] += sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k))))+1] * hamiltonian[(((k) * 7 + (j)))+1]
    sigma_out[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)) + 1))+1] += hamiltonian[((7 * 7 + (i) * 7 + (k)))+1] * sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j)) + 1))+1]
    sigma_out[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)) + 1))+1] -= sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k)) + 1))+1] * hamiltonian[((7 * 7 + (k) * 7 + (j)))+1]
    sigma_out[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j))))+1] += hamiltonian[(((i) * 7 + (k)))+1] * sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j)) + 1))+1]
    sigma_out[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j))))+1] -= sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k))))+1] * hamiltonian[((7 * 7 + (k) * 7 + (j)))+1]
    sigma_out[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j))))+1] += hamiltonian[((7 * 7 + (i) * 7 + (k)))+1] * sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j))))+1]
    sigma_out[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j))))+1] -= sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k)) + 1))+1] * hamiltonian[(((k) * 7 + (j)))+1]
   end
  end
 end
 end
 return
end

@inline function comm_manual_aosoa_constants_direct_prefetch!(sigma_in,sigma_out,hamiltonian,dim::Int32,BX::Int32,BY::Int32,TX::Int32,TY::Int32,BDX::Int32)
 @inbounds begin
 gid = BX * BDX + TX
 for i in Int32(0):Int32(7)-Int32(1)
  j = 0
  for j in Int32(0):Int32(7)-Int32(1)
   for k in Int32(0):Int32(7)-Int32(1)
    sigma_out[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)) + 1))+1] -= hamiltonian[(((i) * 7 + (k)))+1] * sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j))))+1]
    sigma_out[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)) + 1))+1] += sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k))))+1] * hamiltonian[(((k) * 7 + (j)))+1]
    sigma_out[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)) + 1))+1] += hamiltonian[((7 * 7 + (i) * 7 + (k)))+1] * sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j)) + 1))+1]
    sigma_out[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)) + 1))+1] -= sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k)) + 1))+1] * hamiltonian[((7 * 7 + (k) * 7 + (j)))+1]
    sigma_out[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j))))+1] += hamiltonian[(((i) * 7 + (k)))+1] * sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j)) + 1))+1]
    sigma_out[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j))))+1] -= sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k))))+1] * hamiltonian[((7 * 7 + (k) * 7 + (j)))+1]
    sigma_out[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j))))+1] += hamiltonian[((7 * 7 + (i) * 7 + (k)))+1] * sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j))))+1]
    sigma_out[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j))))+1] -= sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k)) + 1))+1] * hamiltonian[(((k) * 7 + (j)))+1]
   end
  end
 end
 end
 return
end

@inline function comm_manual_aosoa_constants_direct_perm!(sigma_in,sigma_out,hamiltonian,dim::Int32,BX::Int32,BY::Int32,TX::Int32,TY::Int32,BDX::Int32)
 @inbounds begin
 gid = BX * BDX + TX
 for i in Int32(0):Int32(7)-Int32(1)
  for k in Int32(0):Int32(7)-Int32(1)
   ham_real_tmp = StudyV4(hamiltonian[(((i) * 7 + (k)))+1])
   ham_imag_tmp = StudyV4(hamiltonian[((7 * 7 + (i) * 7 + (k)))+1])
   sigma_real_tmp = sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k))))+1]
   sigma_imag_tmp = sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k)) + 1))+1]
   for j in Int32(0):Int32(7)-Int32(1)
    sigma_out[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)) + 1))+1] -= ham_real_tmp * sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j))))+1]
    sigma_out[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)) + 1))+1] += sigma_real_tmp * hamiltonian[(((k) * 7 + (j)))+1]
    sigma_out[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)) + 1))+1] += ham_imag_tmp * sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j)) + 1))+1]
    sigma_out[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)) + 1))+1] -= sigma_imag_tmp * hamiltonian[((7 * 7 + (k) * 7 + (j)))+1]
    sigma_out[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j))))+1] += ham_real_tmp * sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j)) + 1))+1]
    sigma_out[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j))))+1] -= sigma_real_tmp * hamiltonian[((7 * 7 + (k) * 7 + (j)))+1]
    sigma_out[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j))))+1] += ham_imag_tmp * sigma_in[(((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j))))+1]
    sigma_out[(((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j))))+1] -= sigma_imag_tmp * hamiltonian[(((k) * 7 + (j)))+1]
   end
  end
 end
 end
 return
end
