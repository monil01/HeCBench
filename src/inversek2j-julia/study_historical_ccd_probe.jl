using Printf
const MAX_LOOP=25
function invkin_cpu(xTgt::Vector{Float32}, yTgt::Vector{Float32})::Vector{Float32}
    n = length(xTgt)
    out = Vector{Float32}(undef, 3 * n)
    for idx in 1:n
        curr_x = xTgt[idx]; curr_y = yTgt[idx]
        a0 = 0.0f0; a1 = 0.0f0; a2 = 0.0f0
        for _ in 1:MAX_LOOP
            for iter in 3:-1:1
                x0 = 0.0f0; y0 = 0.0f0
                x1 = x0 + Float32(cos(a0));        y1 = y0 + Float32(sin(a0))
                x2 = x1 + Float32(cos(a0 + a1));   y2 = y1 + Float32(sin(a0 + a1))
                x3 = x2 + Float32(cos(a0+a1+a2));  y3 = y2 + Float32(sin(a0+a1+a2))
                pe_x = x3; pe_y = y3
                pc_x, pc_y = iter == 1 ? (x0, y0) : iter == 2 ? (x1, y1) : (x2, y2)
                dx1 = pe_x - pc_x; dy1 = pe_y - pc_y
                dx2 = curr_x - pc_x; dy2 = curr_y - pc_y
                l1 = Float32(sqrt(dx1*dx1 + dy1*dy1))
                l2 = Float32(sqrt(dx2*dx2 + dy2*dy2))
                (l1 <= 0.0f0 || l2 <= 0.0f0) && continue
                ax = dx1 / l1; ay = dy1 / l1
                bx = dx2 / l2; by = dy2 / l2
                cv = min(max(ax*bx + ay*by, -1.0f0), 1.0f0)
                sv = ax*by - ay*bx
                ang = Float32(acos(cv))
                sv < 0 && (ang = -ang)
                iter == 1 && (a0 += ang)
                iter == 2 && (a1 += ang)
                iter == 3 && (a2 += ang)
            end
        end
        out[3*(idx-1)+1] = a0
        out[3*(idx-1)+2] = a1
        out[3*(idx-1)+3] = a2
    end
    return out
end

x=Float32[0,1.5,2.5,-1.5,3,2];y=Float32[3,1.5,.5,-1.5,0,0];a=invkin_cpu(x,y)
for j in eachindex(x)
 z=Float64.(a[3j-2:3j]);q=cumsum(z);r=hypot(sum(cos,q)-x[j],sum(sin,q)-y[j]);@printf("HISTORICAL_CCD target=(%.9g,%.9g) radians=(%.9g,%.9g,%.9g) FK_residual=%.17g initial_residual=%.17g\n",x[j],y[j],z...,r,hypot(3-x[j],y[j]))
end
