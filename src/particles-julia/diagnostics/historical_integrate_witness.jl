using CUDA,Printf
const TIMESTEP=Float32(.5)
function integrate_kernel!(pos, vel, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    while i <= n
        base = (i - Int32(1)) * Int32(4)
        @inbounds begin
            vel[base + 2] += Float32(-0.0003) * TIMESTEP
            pos[base + 1] += vel[base + 1] * TIMESTEP
            pos[base + 2] += vel[base + 2] * TIMESTEP
            pos[base + 3] += vel[base + 3] * TIMESTEP
        end
        i += stride
    end
    return
end


root=normpath(joinpath(@__DIR__,".."))
hp=collect(reinterpret(Float32,read(joinpath(root,"study_input_pos_f32.bin"))));hv=collect(reinterpret(Float32,read(joinpath(root,"study_input_vel_f32.bin"))));cpu=collect(reinterpret(Float64,read(joinpath(root,"study_expected_step1_f64.bin"))));dp=CuArray(hp);dv=CuArray(hv);CUDA.device_synchronize(blocking=true);@cuda threads=256 blocks=64 integrate_kernel!(dp,dv,Int32(16384));CUDA.device_synchronize(blocking=true);gotp=Array(dp);gotv=Array(dv);out=vcat(gotp,gotv);write(joinpath(@__DIR__,"historical_integration_only_step1_f32.bin"),out);absolute=maximum(abs.(Float64.(out).-cpu[1:length(out)]));pdiff=maximum(abs.(Float64.(gotp).-cpu[1:65536]));vdiff=maximum(abs.(Float64.(gotv).-cpu[65537:131072]));missing=0.;outside=0;lo=-1f0+.023f0;hi=1f0-.023f0
for i in 1:16384
 b=4(i-1)
 global missing=max(missing,abs(Float64(gotv[b+1])-cpu[65536+b+1]),abs(Float64(gotv[b+3])-cpu[65536+b+3]))
 global outside+=any(x->x<lo||x>hi,gotp[b+1:b+3])
end
finite=all(isfinite,out);failed=absolute>1e-5
@printf("{\"benchmark\":\"particles\",\"diagnostic_only\":true,\"old_kernel_verbatim\":true,\"same_source_fixture\":true,\"steps\":1,\"finite\":%s,\"fails_full_pipeline_contract\":%s,\"max_abs_state_error\":%.12g,\"max_abs_position_error\":%.12g,\"max_abs_velocity_error\":%.12g,\"missing_horizontal_contact_velocity\":%.12g,\"particles_outside_source_cube\":%d,\"study_atol\":1e-5}\n",finite ? "true" : "false",failed ? "true" : "false",absolute,pdiff,vdiff,missing,outside)
finite&&failed&&missing>0&&outside>0||error("Expected source-model distinction not observed")
