using CUDA
const gx=0.0
const gy=0.0
const Re_num=1000.0
const omega=1.7
const mix_param=.9
const dx=1.0/32
const dy=1.0/32
function set_BCs(u,v)
 tx=Int(threadIdx().x)-1;bx=Int(blockIdx().x)-1;by=Int(blockIdx().y)-1;grid_y=Int(gridDim().y)
 @inbounds begin

  ind = (bx * 16) + tx + 1


  
  u[(0)*34+( ind)+1] = 0.0

  v[(0)*34+( ind)+1] = -v[(1)*34+( ind)+1]


  
  u[(32)*34+( ind)+1] = 0.0

  v[(32 + 1)*34+( ind)+1] = -v[(32)*34+( ind)+1]


  
  u[(ind)*34+( 0)+1] = -u[(ind)*34+( 1)+1]

  v[(ind)*34+( 0)+1] = 0.0


  
  u[(ind)*34+( 32 + 1)+1] = 2.0 - u[(ind)*34+( 32)+1]

  v[(ind)*34+( 32)+1] = 0.0


  if (ind == 32) 
    
    u[(0)*34+( 0)+1] = 0.0

    v[(0)*34+( 0)+1] = -v[(1)*34+( 0)+1]

    u[(0)*34+( 32 + 1)+1] = 0.0

    v[(0)*34+( 32 + 1)+1] = -v[(1)*34+( 32 + 1)+1]


    
    u[(32)*34+( 0)+1] = 0.0

    v[(32 + 1)*34+( 0)+1] = -v[(32)*34+( 0)+1]

    u[(32)*34+( 32 + 1)+1] = 0.0

    v[(32 + 1)*34+( 32 + 1)+1] = -v[(32)*34+( 32 + 1)+1]


    
    u[(0)*34+( 0)+1] = -u[(0)*34+( 1)+1]

    v[(0)*34+( 0)+1] = 0.0

    u[(32 + 1)*34+( 0)+1] = -u[(32 + 1)*34+( 1)+1]

    v[(32 + 1)*34+( 0)+1] = 0.0


    
    u[(0)*34+( 32 + 1)+1] = 2.0 - u[(0)*34+( 32)+1]

    v[(0)*34+( 32)+1] = 0.0

    u[(32 + 1)*34+( 32 + 1)+1] = 2.0 - u[(32 + 1)*34+( 32)+1]

    v[(ind)*34+( 32 + 1)+1] = 0.0

  end 


 end
 return
end

function calculate_F(dt::Float64,u,v,F)
 tx=Int(threadIdx().x)-1;bx=Int(blockIdx().x)-1;by=Int(blockIdx().y)-1;grid_y=Int(gridDim().y)
 @inbounds begin
  
  row = (bx * 16) + tx + 1

  col = (by * 1) + 0 + 1


  if (col == 32) 
    
    
    F[(0)*34+( row)+1] = u[(0)*34+( row)+1]

    F[(32)*34+( row)+1] = u[(32)*34+( row)+1]

  else

    
    u_ij = u[(col)*34+( row)+1]

    u_ip1j = u[(col + 1)*34+( row)+1]

    u_ijp1 = u[(col)*34+( row + 1)+1]

    u_im1j = u[(col - 1)*34+( row)+1]

    u_ijm1 = u[(col)*34+( row - 1)+1]


    
    v_ij = v[(col)*34+( row)+1]

    v_ip1j = v[(col + 1)*34+( row)+1]

    v_ijm1 = v[(col)*34+( row - 1)+1]

    v_ip1jm1 = v[(col + 1)*34+( row - 1)+1]


    
    

    du2dx = (((u_ij + u_ip1j) * (u_ij + u_ip1j) - (u_im1j + u_ij) * (u_im1j + u_ij))
        + mix_param * (abs(u_ij + u_ip1j) * (u_ij - u_ip1j)
          - abs(u_im1j + u_ij) * (u_im1j - u_ij))) / (4.0 * dx)

    duvdy = ((v_ij + v_ip1j) * (u_ij + u_ijp1) - (v_ijm1 + v_ip1jm1) * (u_ijm1 + u_ij)
        + mix_param * (abs(v_ij + v_ip1j) * (u_ij - u_ijp1)
          - abs(v_ijm1 + v_ip1jm1) * (u_ijm1 - u_ij))) / (4.0 * dy)

    d2udx2 = (u_ip1j - (2.0 * u_ij) + u_im1j) / (dx * dx)

    d2udy2 = (u_ijp1 - (2.0 * u_ij) + u_ijm1) / (dy * dy)


    F[(col)*34+( row)+1] = u_ij + dt * (((d2udx2 + d2udy2) / Re_num) - du2dx - duvdy + gx)


  end 


 end
 return
end

function calculate_G(dt::Float64,u,v,G)
 tx=Int(threadIdx().x)-1;bx=Int(blockIdx().x)-1;by=Int(blockIdx().y)-1;grid_y=Int(gridDim().y)
 @inbounds begin

  row = (bx * 16) + tx + 1

  col = (by * 1) + 0 + 1


  if (row == 32) 
    
    G[(col)*34+( 0)+1] = v[(col)*34+( 0)+1]

    G[(col)*34+( 32)+1] = v[(col)*34+( 32)+1]


  else

    
    u_ij = u[(col)*34+( row)+1]

    u_ijp1 = u[(col)*34+( row + 1)+1]

    u_im1j = u[(col - 1)*34+( row)+1]

    u_im1jp1 = u[(col - 1)*34+( row + 1)+1]


    
    v_ij = v[(col)*34+( row)+1]

    v_ijp1 = v[(col)*34+( row + 1)+1]

    v_ip1j = v[(col + 1)*34+( row)+1]

    v_ijm1 = v[(col)*34+( row - 1)+1]

    v_im1j = v[(col - 1)*34+( row)+1]


    
    

    dv2dy = ((v_ij + v_ijp1) * (v_ij + v_ijp1) - (v_ijm1 + v_ij) * (v_ijm1 + v_ij)
        + mix_param * (abs(v_ij + v_ijp1) * (v_ij - v_ijp1)
          - abs(v_ijm1 + v_ij) * (v_ijm1 - v_ij))) / (4.0 * dy)

    duvdx = ((u_ij + u_ijp1) * (v_ij + v_ip1j) - (u_im1j + u_im1jp1) * (v_im1j + v_ij)
        + mix_param * (abs(u_ij + u_ijp1) * (v_ij - v_ip1j) 
          - abs(u_im1j + u_im1jp1) * (v_im1j - v_ij))) / (4.0 * dx)

    d2vdx2 = (v_ip1j - (2.0 * v_ij) + v_im1j) / (dx * dx)

    d2vdy2 = (v_ijp1 - (2.0 * v_ij) + v_ijm1) / (dy * dy)


    G[(col)*34+( row)+1] = v_ij + dt * (((d2vdx2 + d2vdy2) / Re_num) - dv2dy - duvdx + gy)


  end 


 end
 return
end

function sum_pressure(pres_red,pres_black,pres_sum)
 tx=Int(threadIdx().x)-1;bx=Int(blockIdx().x)-1;by=Int(blockIdx().y)-1;grid_y=Int(gridDim().y)
 @inbounds begin

  row = (bx * 16) + tx + 1

  col = (by * 1) + 0 + 1


  
  sum_cache = CUDA.CuStaticSharedArray(Float64,(16,))


  # NUM_2 is compile-time16


  pres_r = pres_red[(col)*18+( row)+1]

  pres_b = pres_black[(col)*18+( row)+1]


  
  sum_cache[tx+1] = (pres_r * pres_r) + (pres_b * pres_b)


  
  CUDA.sync_threads()


  
  i = 16 >> 1

  while (i != 0) 
    if (tx < i) 
      sum_cache[tx+1] += sum_cache[tx + i+1]

    end
    CUDA.sync_threads()

    i >>= 1

  end

  
  if (tx == 0) 
    pres_sum[by + (grid_y * bx)+1] = sum_cache[0+1]

  end


 end
 return
end

function set_horz_pres_BCs(pres_red,pres_black)
 tx=Int(threadIdx().x)-1;bx=Int(blockIdx().x)-1;by=Int(blockIdx().y)-1;grid_y=Int(gridDim().y)
 @inbounds begin

  col = (bx * 16) + tx + 1

  col = (col * 2) - 1


  # NUM_2 is compile-time16


  
  pres_black[(col)*18+( 0)+1] = pres_red[(col)*18+( 1)+1]

  pres_red[(col + 1)*18+( 0)+1] = pres_black[(col + 1)*18+( 1)+1]


  
  pres_red[(col)*18+( 16 + 1)+1] = pres_black[(col)*18+( 16)+1]

  pres_black[(col + 1)*18+( 16 + 1)+1] = pres_red[(col + 1)*18+( 16)+1]



 end
 return
end

function set_vert_pres_BCs(pres_red,pres_black)
 tx=Int(threadIdx().x)-1;bx=Int(blockIdx().x)-1;by=Int(blockIdx().y)-1;grid_y=Int(gridDim().y)
 @inbounds begin

  row = (bx * 16) + tx + 1


  # NUM_2 is compile-time16


  
  pres_black[(0)*18+( row)+1] = pres_red[(1)*18+( row)+1]

  pres_red[(0)*18+( row)+1] = pres_black[(1)*18+( row)+1]


  
  pres_black[(32 + 1)*18+( row)+1] = pres_red[(32)*18+( row)+1]

  pres_red[(32 + 1)*18+( row)+1] = pres_black[(32)*18+( row)+1]



 end
 return
end

function red_kernel(dt::Float64,F,G,pres_black,pres_red)
 tx=Int(threadIdx().x)-1;bx=Int(blockIdx().x)-1;by=Int(blockIdx().y)-1;grid_y=Int(gridDim().y)
 @inbounds begin

  row = (bx * 16) + tx + 1

  col = (by * 1) + 0 + 1


  # NUM_2 is compile-time16
      

  p_ij = pres_red[(col)*18+( row)+1]


  p_im1j = pres_black[(col - 1)*18+( row)+1]

  p_ip1j = pres_black[(col + 1)*18+( row)+1]

  p_ijm1 = pres_black[(col)*18+( row - (col & 1))+1]

  p_ijp1 = pres_black[(col)*18+( row + ((col + 1) & 1))+1]


  
  rhs = (((F[(col)*34+( (2 * row) - (col & 1))+1]
          - F[(col - 1)*34+( (2 * row) - (col & 1))+1]) / dx)
      + ((G[(col)*34+( (2 * row) - (col & 1))+1]
          - G[(col)*34+( (2 * row) - (col & 1) - 1)+1]) / dy)) / dt


  pres_red[(col)*18+( row)+1] = p_ij * (1.0 - omega) + omega * 
    (((p_ip1j + p_im1j) / (dx * dx)) + ((p_ijp1 + p_ijm1) / (dy * dy)) - 
     rhs) / ((2.0 / (dx * dx)) + (2.0 / (dy * dy)))



 end
 return
end

function black_kernel(dt::Float64,F,G,pres_red,pres_black)
 tx=Int(threadIdx().x)-1;bx=Int(blockIdx().x)-1;by=Int(blockIdx().y)-1;grid_y=Int(gridDim().y)
 @inbounds begin

  row = (bx * 16) + tx + 1

  col = (by * 1) + 0 + 1


  # NUM_2 is compile-time16


  p_ij = pres_black[(col)*18+( row)+1]


  p_im1j = pres_red[(col - 1)*18+( row)+1]

  p_ip1j = pres_red[(col + 1)*18+( row)+1]

  p_ijm1 = pres_red[(col)*18+( row - ((col + 1) & 1))+1]

  p_ijp1 = pres_red[(col)*18+( row + (col & 1))+1]


  
  rhs = (((F[(col)*34+( (2 * row) - ((col + 1) & 1))+1]
          - F[(col - 1)*34+( (2 * row) - ((col + 1) & 1))+1]) / dx)
      + ((G[(col)*34+( (2 * row) - ((col + 1) & 1))+1]
          - G[(col)*34+( (2 * row) - ((col + 1) & 1) - 1)+1]) / dy)) / dt


  pres_black[(col)*18+( row)+1] = p_ij * (1.0 - omega) + omega * 
    (((p_ip1j + p_im1j) / (dx * dx)) + ((p_ijp1 + p_ijm1) / (dy * dy)) - 
     rhs) / ((2.0 / (dx * dx)) + (2.0 / (dy * dy)))



 end
 return
end

function calc_residual(dt::Float64,F,G,pres_red,pres_black,res_array)
 tx=Int(threadIdx().x)-1;bx=Int(blockIdx().x)-1;by=Int(blockIdx().y)-1;grid_y=Int(gridDim().y)
 @inbounds begin

  row = (bx * 16) + tx + 1

  col = (by * 1) + 0 + 1


  # NUM_2 is compile-time16


  

  
  p_ij = pres_red[(col)*18+( row)+1]


  p_im1j = pres_black[(col - 1)*18+( row)+1]

  p_ip1j = pres_black[(col + 1)*18+( row)+1]

  p_ijm1 = pres_black[(col)*18+( row - (col & 1))+1]

  p_ijp1 = pres_black[(col)*18+( row + ((col + 1) & 1))+1]


  rhs = (((F[(col)*34+( (2 * row) - (col & 1))+1] - F[(col - 1)*34+( (2 * row) - (col & 1))+1]) / dx)
      +  ((G[(col)*34+( (2 * row) - (col & 1))+1] - G[(col)*34+( (2 * row) - (col & 1) - 1)+1]) / dy)) / dt


  
  res = ((p_ip1j - (2.0 * p_ij) + p_im1j) / (dx * dx))
    + ((p_ijp1 - (2.0 * p_ij) + p_ijm1) / (dy * dy)) - rhs


  
  p_ij = pres_black[(col)*18+( row)+1]


  p_im1j = pres_red[(col - 1)*18+( row)+1]

  p_ip1j = pres_red[(col + 1)*18+( row)+1]

  p_ijm1 = pres_red[(col)*18+( row - ((col + 1) & 1))+1]

  p_ijp1 = pres_red[(col)*18+( row + (col & 1))+1]


  
  rhs = (((F[(col)*34+( (2 * row) - ((col + 1) & 1))+1] - F[(col - 1)*34+( (2 * row) - ((col + 1) & 1))+1]) / dx)
      +  ((G[(col)*34+( (2 * row) - ((col + 1) & 1))+1] - G[(col)*34+( (2 * row) - ((col + 1) & 1) - 1)+1]) / dy)) / dt


  
  res2 = ((p_ip1j - (2.0 * p_ij) + p_im1j) / (dx * dx))
    + ((p_ijp1 - (2.0 * p_ij) + p_ijm1) / (dy * dy)) - rhs


  
  sum_cache = CUDA.CuStaticSharedArray(Float64,(16,))


  sum_cache[tx+1] = (res * res) + (res2 * res2)


  
  CUDA.sync_threads()


  
  i = 16 >> 1

  while (i != 0) 
    if (tx < i) 
      sum_cache[tx+1] += sum_cache[tx + i+1]

    end
    CUDA.sync_threads()

    i >>= 1

  end

  
  if (tx == 0) 
    res_array[by + (grid_y * bx)+1] = sum_cache[0+1]

  end

 end
 return
end

function calculate_u(dt::Float64,F,pres_red,pres_black,u,max_u)
 tx=Int(threadIdx().x)-1;bx=Int(blockIdx().x)-1;by=Int(blockIdx().y)-1;grid_y=Int(gridDim().y)
 @inbounds begin

  row = (bx * 16) + tx + 1

  col = (by * 1) + 0 + 1


  
  max_cache = CUDA.CuStaticSharedArray(Float64,(16,))

  max_cache[tx+1] = 0.0


  # NUM_2 is compile-time16

  new_u = 0.0


  if (col != 32) 

    

    
    p_ij = pres_red[(col)*18+( row)+1]

    p_ip1j = pres_black[(col + 1)*18+( row)+1]


    new_u = F[(col)*34+( (2 * row) - (col & 1))+1] - (dt * (p_ip1j - p_ij) / dx)

    u[(col)*34+( (2 * row) - (col & 1))+1] = new_u


    
    p_ij = pres_black[(col)*18+( row)+1]

    p_ip1j = pres_red[(col + 1)*18+( row)+1]


    new_u2 = F[(col)*34+( (2 * row) - ((col + 1) & 1))+1] - (dt * (p_ip1j - p_ij) / dx)

    u[(col)*34+( (2 * row) - ((col + 1) & 1))+1] = new_u2


    
    new_u = max(abs(new_u), abs(new_u2))


    if ((2 * row) == 32) 
      
      new_u = max(new_u, abs( u[(col)*34+( 32 + 1)+1] ))

    end
  else
    
    new_u = max(abs( u[(32)*34+( (2 * row))+1] ), abs( u[(0)*34+( (2 * row))+1] ))

    new_u = max(abs( u[(32)*34+( (2 * row) - 1)+1] ), new_u)

    new_u = max(abs( u[(0)*34+( (2 * row) - 1)+1] ), new_u)


    new_u = max(abs( u[(32 + 1)*34+( (2 * row))+1] ), new_u)

    new_u = max(abs( u[(32 + 1)*34+( (2 * row) - 1)+1] ), new_u)


  end 

  
  max_cache[tx+1] = new_u


  
  CUDA.sync_threads()


  
  i = 16 >> 1

  while (i != 0) 
    if (tx < i) 
      max_cache[tx+1] = max(max_cache[tx+1], max_cache[tx + i+1])

    end
    CUDA.sync_threads()

    i >>= 1

  end

  
  if (tx == 0) 
    max_u[by + (grid_y * bx)+1] = max_cache[0+1]

  end

 end
 return
end

function calculate_v(dt::Float64,G,pres_red,pres_black,v,max_v)
 tx=Int(threadIdx().x)-1;bx=Int(blockIdx().x)-1;by=Int(blockIdx().y)-1;grid_y=Int(gridDim().y)
 @inbounds begin

  row = (bx * 16) + tx + 1

  col = (by * 1) + 0 + 1


  
  max_cache = CUDA.CuStaticSharedArray(Float64,(16,))

  max_cache[tx+1] = 0.0


  # NUM_2 is compile-time16

  new_v = 0.0


  if (row != 16) 
    

    
    p_ij = pres_red[(col)*18+( row)+1]

    p_ijp1 = pres_black[(col)*18+( row + ((col + 1) & 1))+1]


    new_v = G[(col)*34+( (2 * row) - (col & 1))+1] - (dt * (p_ijp1 - p_ij) / dy)

    v[(col)*34+( (2 * row) - (col & 1))+1] = new_v


    
    p_ij = pres_black[(col)*18+( row)+1]

    p_ijp1 = pres_red[(col)*18+( row + (col & 1))+1]


    new_v2 = G[(col)*34+( (2 * row) - ((col + 1) & 1))+1] - (dt * (p_ijp1 - p_ij) / dy)

    v[(col)*34+( (2 * row) - ((col + 1) & 1))+1] = new_v2


    
    new_v = max(abs(new_v), abs(new_v2))


    if (col == 32) 
      
      new_v = max(new_v, abs( v[(32 + 1)*34+( (2 * row))+1] ))

    end

  else

    if ((col & 1) == 1) 
      
      p_ij = pres_red[(col)*18+( row)+1]

      p_ijp1 = pres_black[(col)*18+( row + ((col + 1) & 1))+1]


      new_v = G[(col)*34+( (2 * row) - (col & 1))+1] - (dt * (p_ijp1 - p_ij) / dy)

      v[(col)*34+( (2 * row) - (col & 1))+1] = new_v

    else
      
      p_ij = pres_black[(col)*18+( row)+1]

      p_ijp1 = pres_red[(col)*18+( row + (col & 1))+1]


      new_v = G[(col)*34+( (2 * row) - ((col + 1) & 1))+1] - (dt * (p_ijp1 - p_ij) / dy)

      v[(col)*34+( (2 * row) - ((col + 1) & 1))+1] = new_v

    end

    
    new_v = abs(new_v)


    
    new_v = max(abs( v[(col)*34+( 32)+1] ), new_v)

    new_v = max(abs( v[(col)*34+( 0)+1] ), new_v)


    new_v = max(abs( v[(col)*34+( 32 + 1)+1] ), new_v)

  end 

  
  max_cache[tx+1] = new_v


  
  CUDA.sync_threads()


  
  i = 16 >> 1

  while (i != 0) 
    if (tx < i) 
      max_cache[tx+1] = max(max_cache[tx+1], max_cache[tx + i+1])

    end
    CUDA.sync_threads()

    i >>= 1

  end

  
  if (tx == 0) 
    max_v[by + (grid_y * bx)+1] = max_cache[0+1]

  end

 end
 return
end
