@inline function floyd_update!(x,y,distance,path,n,k)
    if x<n&&y<n
        @inbounds old=Int32(distance[y*n+x+Int32(1)]);candidate=Int32(distance[y*n+k+Int32(1)]+distance[k*n+x+Int32(1)])
        if candidate<old
            @inbounds distance[y*n+x+Int32(1)]=UInt32(candidate);path[y*n+x+Int32(1)]=UInt32(k)
        end
    end
    nothing
end
