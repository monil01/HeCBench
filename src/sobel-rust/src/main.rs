// Sobel filter, RGBA uchar4 in / uchar4 out, ported from sobel-cuda.
// Uses a synthetic random 512x512 image so we do not need to depend on the
// AMD SDK bitmap helper. Correctness is checked against a CPU reference on
// the same input.
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::env;
use std::time::Instant;

const KERNEL_SRC: &str = r#"
extern "C" __global__ void sobel_filter(const uchar4* __restrict__ inputImage,
                                        uchar4* __restrict__ outputImage,
                                        const unsigned int width,
                                        const unsigned int height)
{
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= 1 && x < (width - 1) && y >= 1 && y < height - 1) {
        int c = x + y * width;
        auto cf4 = [](uchar4 v){
            float4 r; r.x=v.x; r.y=v.y; r.z=v.z; r.w=v.w; return r;
        };
        float4 i00 = cf4(inputImage[c - 1 - width]);
        float4 i01 = cf4(inputImage[c - width]);
        float4 i02 = cf4(inputImage[c + 1 - width]);
        float4 i10 = cf4(inputImage[c - 1]);
        float4 i12 = cf4(inputImage[c + 1]);
        float4 i20 = cf4(inputImage[c - 1 + width]);
        float4 i21 = cf4(inputImage[c + width]);
        float4 i22 = cf4(inputImage[c + 1 + width]);

        float Gx_x = i00.x + 2.f*i10.x + i20.x - i02.x - 2.f*i12.x - i22.x;
        float Gx_y = i00.y + 2.f*i10.y + i20.y - i02.y - 2.f*i12.y - i22.y;
        float Gx_z = i00.z + 2.f*i10.z + i20.z - i02.z - 2.f*i12.z - i22.z;
        float Gx_w = i00.w + 2.f*i10.w + i20.w - i02.w - 2.f*i12.w - i22.w;

        float Gy_x = i00.x - i20.x + 2.f*i01.x - 2.f*i21.x + i02.x - i22.x;
        float Gy_y = i00.y - i20.y + 2.f*i01.y - 2.f*i21.y + i02.y - i22.y;
        float Gy_z = i00.z - i20.z + 2.f*i01.z - 2.f*i21.z + i02.z - i22.z;
        float Gy_w = i00.w - i20.w + 2.f*i01.w - 2.f*i21.w + i02.w - i22.w;

        auto clamp = [](float v){
            v = v > 255.f ? 255.f : v;
            v = v < 0.f ? 0.f : v;
            return (unsigned char)v;
        };
        uchar4 o;
        o.x = clamp(sqrtf(Gx_x*Gx_x + Gy_x*Gy_x)/2.f);
        o.y = clamp(sqrtf(Gx_y*Gx_y + Gy_y*Gy_y)/2.f);
        o.z = clamp(sqrtf(Gx_z*Gx_z + Gy_z*Gy_z)/2.f);
        o.w = clamp(sqrtf(Gx_w*Gx_w + Gy_w*Gy_w)/2.f);
        outputImage[c] = o;
    }
}
"#;

fn reference(out: &mut [u8], input: &[u8], w: usize, h: usize) {
    for y in 1..h-1 {
        for x in 1..w-1 {
            let c = (x + y * w) * 4;
            let get = |dx: i32, dy: i32, ch: usize| -> f32 {
                let idx = (((y as i32 + dy) as usize) * w + (x as i32 + dx) as usize) * 4 + ch;
                input[idx] as f32
            };
            for ch in 0..4 {
                let gx = get(-1,-1,ch) + 2.0*get(-1,0,ch) + get(-1,1,ch)
                       - get(1,-1,ch) - 2.0*get(1,0,ch) - get(1,1,ch);
                let gy = get(-1,-1,ch) - get(-1,1,ch) + 2.0*get(0,-1,ch)
                       - 2.0*get(0,1,ch) + get(1,-1,ch) - get(1,1,ch);
                let mag = (gx*gx + gy*gy).sqrt() / 2.0;
                let v = mag.max(0.0).min(255.0);
                out[c + ch] = v as u8;
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let w: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(512);
    let h: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(512);
    let iters: i32 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(100);

    let n_bytes = w * h * 4;
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let input: Vec<u8> = (0..n_bytes).map(|_| rng.gen_range(0..=255u8)).collect();

    let dev = CudaDevice::new(0)?;
    let ptx = compile_ptx(KERNEL_SRC)?;
    dev.load_ptx(ptx, "sobel", &["sobel_filter"])?;
    let f = dev.get_func("sobel", "sobel_filter").unwrap();

    let d_in = dev.htod_copy(input.clone())?;
    let mut d_out = dev.alloc_zeros::<u8>(n_bytes)?;

    let cfg = LaunchConfig {
        grid_dim: ((w as u32) / 16, (h as u32) / 16, 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };

    dev.synchronize()?;
    let start = Instant::now();
    for _ in 0..iters {
        unsafe { f.clone().launch(cfg, (&d_in, &mut d_out, w as u32, h as u32))?; }
    }
    dev.synchronize()?;
    let elapsed = start.elapsed();
    println!("Average kernel execution time: {:.3} (us)",
             elapsed.as_micros() as f64 / iters as f64);

    let gpu_out: Vec<u8> = dev.dtoh_sync_copy(&d_out)?;
    let mut ref_out = vec![0u8; n_bytes];
    reference(&mut ref_out, &input, w, h);

    // Normalised L2 error, per the CUDA sample's compare()
    let mut err = 0f64;
    let mut refn = 0f64;
    for i in 1..n_bytes {
        let d = gpu_out[i] as f64 - ref_out[i] as f64;
        err += d * d;
        refn += (ref_out[i] as f64) * (ref_out[i] as f64);
    }
    let rel = err.sqrt() / refn.sqrt().max(1e-9);
    println!("Relative L2 error = {:.3e}", rel);
    println!("{}", if rel < 1e-6 { "PASS" } else { "FAIL" });
    Ok(())
}
