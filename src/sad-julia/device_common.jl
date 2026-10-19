# Full valid-window RGB L1 operator, translating the explicitly corrected native
# coordinate bounds/stride. Native requests unroll4; Julia uses ordinary typed
# loops, so compiler unrolling policy remains a disclosed performance covariate.
@inline function compute_pixel!(sad,image,kernel,w,h,kw,kh,col,row)
    one=Int32(1);three=Int32(3);guard=Int32(17)
    vw=w-kw+one;vh=h-kh+one
    if row<vh && col<vw
        total=Int32(0)
        @inbounds for kr=Int32(0):kh-one
            for kc=Int32(0):kw-one
                ia=((row+kr)*w+col+kc)*three+guard
                ka=(kr*kw+kc)*three+guard
                total+=abs(Int32(image[ia])-Int32(kernel[ka]))+abs(Int32(image[ia+one])-Int32(kernel[ka+one]))+abs(Int32(image[ia+Int32(2)])-Int32(kernel[ka+Int32(2)]))
            end
        end
        value=unsafe_trunc(Int32,Float32(total)/Float32(kw*kh))
        @inbounds sad[row*vw+col+guard]=value
    end
    nothing
end
@inline function local_minimum(sad,n,gid,stride)
    temp=Int32(5000);offset=Int32(0)
    @inbounds while gid+offset<n
        temp=min(temp,sad[gid+offset+Int32(17)]);offset+=stride
    end
    temp
end
