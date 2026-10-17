using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function softmax_jacobian_ka!(input, jac, dim::Int32,::Val{DIM}) where {DIM}
    smem = @localmem Float32 (DIM+256,)
    y = smem
    red_offset = Int(dim)
    b0 = Int32(@index(Group,Linear)) - Int32(1)
    tid0 = Int32(@index(Local,Linear)) - Int32(1)
    nt = Int32(256)

    lmax = -Inf32
    i = tid0
    while i < dim
        @inbounds v = input[Int(b0 * dim + i) + 1]
        lmax = max(lmax, v)
        i += nt
    end
    @inbounds smem[red_offset + Int(tid0) + 1] = lmax
    @synchronize

    stride = nt >>> Int32(1)
    while stride > 0
        if tid0 < stride
            @inbounds smem[red_offset + Int(tid0) + 1] =
                max(smem[red_offset + Int(tid0) + 1], smem[red_offset + Int(tid0 + stride) + 1])
        end
        @synchronize
        stride >>>= Int32(1)
    end
    @inbounds maxv = smem[red_offset + 1]
    @synchronize

    lsum = Float32(0)
    i = tid0
    while i < dim
        @inbounds e = exp(input[Int(b0 * dim + i) + 1] - maxv)
        @inbounds y[Int(i) + 1] = e
        lsum += e
        i += nt
    end
    @inbounds smem[red_offset + Int(tid0) + 1] = lsum
    @synchronize

    stride = nt >>> Int32(1)
    while stride > 0
        if tid0 < stride
            @inbounds smem[red_offset + Int(tid0) + 1] += smem[red_offset + Int(tid0 + stride) + 1]
        end
        @synchronize
        stride >>>= Int32(1)
    end
    @inbounds inv = Float32(1) / smem[red_offset + 1]
    @synchronize

    i = tid0
    while i < dim
        @inbounds y[Int(i) + 1] *= inv
        i += nt
    end
    @synchronize

    total = Int64(dim) * Int64(dim)
    idx = Int64(tid0)
    base = Int64(b0) * total
    while idx < total
        row = Int32(idx ÷ Int64(dim))
        col = Int32(idx - Int64(row) * Int64(dim))
        @inbounds yi = y[Int(row) + 1]
        @inbounds yj = y[Int(col) + 1]
        @inbounds jac[Int(base + idx) + 1] = yi * ((row == col ? Float32(1) : Float32(0)) - yj)
        idx += Int64(nt)
    end
end

@inline function jac_warp_max(v)
 for offset in (16,8,4,2,1);v=max(v,CUDA.shfl_down_sync(UInt32(0xffffffff),v,offset));end
 return v
end
@inline function jac_warp_sum(v)
 for offset in (16,8,4,2,1);v+=CUDA.shfl_down_sync(UInt32(0xffffffff),v,offset);end
 return v
end
@kernel function softmax_jacobian_ka2!(input,jac,dim::Int32,::Val{DIM}) where {DIM}
 smem=@localmem Float32 (DIM+256,)
 b0=Int32(@index(Group,Linear))-Int32(1);tid0=Int32(@index(Local,Linear))-Int32(1);nt=Int32(256)
 lane=tid0&Int32(31);warp=tid0>>>Int32(5);nwarps=Int32(8)
 lmax=-Inf32;i=tid0
 while i<dim
  @inbounds lmax=max(lmax,input[b0*dim+i+Int32(1)]);i+=nt
 end
 lmax=jac_warp_max(lmax)
 if lane==0;@inbounds smem[dim+warp+Int32(1)]=lmax;end
 @synchronize
 if warp==0
  value=lane<nwarps ? (@inbounds smem[dim+lane+Int32(1)]) : -Inf32
  value=jac_warp_max(value)
  if lane==0;@inbounds smem[dim+Int32(1)]=value;end
 end
 @synchronize
 maxv=@inbounds smem[dim+Int32(1)]
 # Protect the scalar-read lifetime before the next phase reuses warp partials.
 @synchronize
 lsum=0.f0;i=tid0
 while i<dim
  e=exp((@inbounds input[b0*dim+i+Int32(1)])-maxv);@inbounds smem[i+Int32(1)]=e;lsum+=e;i+=nt
 end
 lsum=jac_warp_sum(lsum)
 if lane==0;@inbounds smem[dim+warp+Int32(1)]=lsum;end
 @synchronize
 if warp==0
  value=lane<nwarps ? (@inbounds smem[dim+lane+Int32(1)]) : 0.f0
  value=jac_warp_sum(value)
  if lane==0;@inbounds smem[dim+Int32(1)]=value;end
 end
 @synchronize
 inv=1.f0/(@inbounds smem[dim+Int32(1)]);i=tid0
 while i<dim;@inbounds smem[i+Int32(1)]*=inv;i+=nt;end
 @synchronize
 if (dim&Int32(3))==0
  perrow=dim>>>Int32(2);total=dim*perrow;q=tid0;base=b0*dim*dim
  while q<total
   row=div(q,perrow);col=(q-row*perrow)*Int32(4);yi=@inbounds smem[row+Int32(1)]
   for k in Int32(0):Int32(3)
    yj=@inbounds smem[col+k+Int32(1)];@inbounds jac[base+row*dim+col+k+Int32(1)]=yi*((row==col+k ? 1.f0 : 0.f0)-yj)
   end
   q+=nt
  end
 else
  total=dim*dim;index=tid0;base=b0*total
  while index<total
   row=div(index,dim);col=index-row*dim;@inbounds jac[base+index+Int32(1)]=smem[row+Int32(1)]*((row==col ? 1.f0 : 0.f0)-smem[col+Int32(1)]);index+=nt
  end
 end
end
function factory(input,out,dim,batch,mode)
 k=mode=="k1" ? softmax_jacobian_ka! : softmax_jacobian_ka2!
 launch=k(get_backend(input),256);tag=Val(dim)
 return ()->launch(input,out,Int32(dim),tag;ndrange=batch*256)
end
jacobian_study(factory)
