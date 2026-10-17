using CUDA
using JACC
include(joinpath(@__DIR__,"..","s8n_distances.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function cube_select_kernel!(index,n::Int32, radius::Int32, xyz, idx_out)
    batch_idx = (Int32(index)-Int32(1)) ÷ Int32(512)
    base_xyz = batch_idx * n * Int32(3)
    base_out = batch_idx * n * Int32(8)
    i = (Int32(index)-Int32(1)) % Int32(512)
    while i < n
        x = @inbounds xyz[base_xyz + i * Int32(3) + Int32(1)]
        y = @inbounds xyz[base_xyz + i * Int32(3) + Int32(2)]
        z = @inbounds xyz[base_xyz + i * Int32(3) + Int32(3)]
        temp_dist = init_dist8()
        for k in Int32(0):Int32(7)
            @inbounds idx_out[base_out + i * Int32(8) + k + Int32(1)] = i
        end
        for j in Int32(0):(n - Int32(1))
            j == i && continue
            tx = @inbounds xyz[base_xyz + j * Int32(3) + Int32(1)]
            ty = @inbounds xyz[base_xyz + j * Int32(3) + Int32(2)]
            tz = @inbounds xyz[base_xyz + j * Int32(3) + Int32(3)]
            dist = (x - tx) * (x - tx) + (y - ty) * (y - ty) + (z - tz) * (z - tz)
            dist > radius && continue
            temp_idx = (tx > x ? Int32(4) : Int32(0)) + (ty > y ? Int32(2) : Int32(0)) + (tz > z ? Int32(1) : Int32(0))
            slot = Int(temp_idx + Int32(1))
            if dist < temp_dist[slot]
                @inbounds idx_out[base_out + i * Int32(8) + temp_idx + Int32(1)] = j
                temp_dist[slot] = dist
            end
        end
        i += Int32(512)
    end
    nothing
end

function cube_select_two_kernel!(index,n::Int32, radius::Int32, xyz, idx_out)
    batch_idx = (Int32(index)-Int32(1)) ÷ Int32(512)
    base_xyz = batch_idx * n * Int32(3)
    base_out = batch_idx * n * Int32(16)
    i = (Int32(index)-Int32(1)) % Int32(512)
    while i < n
        x = @inbounds xyz[base_xyz + i * Int32(3) + Int32(1)]
        y = @inbounds xyz[base_xyz + i * Int32(3) + Int32(2)]
        z = @inbounds xyz[base_xyz + i * Int32(3) + Int32(3)]
        temp_dist = init_dist16()
        for k in Int32(0):Int32(15)
            @inbounds idx_out[base_out + i * Int32(16) + k + Int32(1)] = i
        end
        for j in Int32(0):(n - Int32(1))
            j == i && continue
            tx = @inbounds xyz[base_xyz + j * Int32(3) + Int32(1)]
            ty = @inbounds xyz[base_xyz + j * Int32(3) + Int32(2)]
            tz = @inbounds xyz[base_xyz + j * Int32(3) + Int32(3)]
            dist = (x - tx) * (x - tx) + (y - ty) * (y - ty) + (z - tz) * (z - tz)
            dist > radius && continue
            temp_idx = (tx > x ? Int32(8) : Int32(0)) + (ty > y ? Int32(4) : Int32(0)) + (tz > z ? Int32(2) : Int32(0))
            flag = false
            for k in Int32(0):Int32(1)
                slot = Int(temp_idx + k + Int32(1))
                if dist < temp_dist[slot]
                    flag = true
                end
                if flag
                    for kk in Int32(1):-Int32(1):(k + Int32(1))
                        @inbounds idx_out[base_out + i * Int32(16) + temp_idx + kk + Int32(1)] =
                            idx_out[base_out + i * Int32(16) + temp_idx + kk]
                        temp_dist[Int(temp_idx + kk + Int32(1))] = temp_dist[Int(temp_idx + kk)]
                    end
                    @inbounds idx_out[base_out + i * Int32(16) + temp_idx + k + Int32(1)] = j
                    temp_dist[slot] = dist
                    break
                end
            end
        end
        i += Int32(512)
    end
    nothing
end

function cube_select_four_kernel!(index,n::Int32, radius::Int32, xyz, idx_out)
    batch_idx = (Int32(index)-Int32(1)) ÷ Int32(512)
    base_xyz = batch_idx * n * Int32(3)
    base_out = batch_idx * n * Int32(32)
    i = (Int32(index)-Int32(1)) % Int32(512)
    while i < n
        x = @inbounds xyz[base_xyz + i * Int32(3) + Int32(1)]
        y = @inbounds xyz[base_xyz + i * Int32(3) + Int32(2)]
        z = @inbounds xyz[base_xyz + i * Int32(3) + Int32(3)]
        temp_dist = init_dist32()
        for k in Int32(0):Int32(31)
            @inbounds idx_out[base_out + i * Int32(32) + k + Int32(1)] = i
        end
        for j in Int32(0):(n - Int32(1))
            j == i && continue
            tx = @inbounds xyz[base_xyz + j * Int32(3) + Int32(1)]
            ty = @inbounds xyz[base_xyz + j * Int32(3) + Int32(2)]
            tz = @inbounds xyz[base_xyz + j * Int32(3) + Int32(3)]
            dist = (x - tx) * (x - tx) + (y - ty) * (y - ty) + (z - tz) * (z - tz)
            dist > radius && continue
            temp_idx = (tx > x ? Int32(16) : Int32(0)) + (ty > y ? Int32(8) : Int32(0)) + (tz > z ? Int32(4) : Int32(0))
            flag = false
            for k in Int32(0):Int32(3)
                slot = Int(temp_idx + k + Int32(1))
                if dist < temp_dist[slot]
                    flag = true
                end
                if flag
                    for kk in Int32(3):-Int32(1):(k + Int32(1))
                        @inbounds idx_out[base_out + i * Int32(32) + temp_idx + kk + Int32(1)] =
                            idx_out[base_out + i * Int32(32) + temp_idx + kk]
                        temp_dist[Int(temp_idx + kk + Int32(1))] = temp_dist[Int(temp_idx + kk)]
                    end
                    @inbounds idx_out[base_out + i * Int32(32) + temp_idx + k + Int32(1)] = j
                    temp_dist[slot] = dist
                    break
                end
            end
        end
        i += Int32(512)
    end
    nothing
end

function study_prepare(xyz,out,b,n,ranks)
    k=ranks==1 ? cube_select_kernel! : ranks==2 ? cube_select_two_kernel! : cube_select_four_kernel!;backend=JACC.get_backend(:cuda);spec=JACC.LaunchSpec{typeof(backend)}(;threads=512,blocks=b,shmem_size=0,sync=false)
    ()->JACC.parallel_for(k,spec,b*512,Int32(n),Int32(512),xyz,out)
end
exit(study_main(study_prepare))
