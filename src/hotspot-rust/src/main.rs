// hotspot-rust: 5-point stencil update with synthetic deterministic input.
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::time::Instant;

const L: usize = 512;
const ITERS: i32 = 200;

const KERNEL_SRC: &str = r#"
#define L 512
extern "C" __global__ void hotspot(
    float* out, const float* cur, const float* power,
    float step_div_Cap, float Rx_1, float Ry_1, float Rz_1)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= L || y >= L) return;
    int N = y - 1; if (N < 0) N = 0;
    int S = y + 1; if (S > L - 1) S = L - 1;
    int W = x - 1; if (W < 0) W = 0;
    int E = x + 1; if (E > L - 1) E = L - 1;
    int idx = y * L + x;
    float t = cur[idx];
    out[idx] = t + step_div_Cap * (power[idx]
      + (cur[S*L + x] + cur[N*L + x] - 2.f*t) * Ry_1
      + (cur[y*L + E] + cur[y*L + W] - 2.f*t) * Rx_1
      + (80.f - t) * Rz_1);
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let l2 = L * L;
    let mut s: u64 = 20250723;
    let mut tvec = vec![0f32; l2];
    let mut pvec = vec![0f32; l2];
    for i in 0..l2 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        tvec[i] = 300.0 + (((s >> 33) & 0x7fffffff) as f32) / 0x7fffffff as f32 * 50.0;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        pvec[i] = (((s >> 33) & 0x7fffffff) as f32) / 0x7fffffff as f32 * 0.5;
    }
    let chip_height = 0.016f32; let chip_width = 0.016f32;
    let grid_h = chip_height / L as f32; let grid_w = chip_width / L as f32;
    let t_chip = 0.0005f32; let k_si = 100f32; let spec_heat_si = 1.75e6f32;
    let factor_chip = 0.5f32; let max_pd = 3.0e6f32; let precision = 0.001f32;
    let cap = factor_chip * spec_heat_si * t_chip * grid_w * grid_h;
    let rx = grid_w / (2.0 * k_si * t_chip * grid_h);
    let ry = grid_h / (2.0 * k_si * t_chip * grid_w);
    let rz = t_chip / (k_si * grid_h * grid_w);
    let step = precision / (max_pd / (factor_chip * t_chip * spec_heat_si));
    let step_div_cap = step / cap;
    let rx_1 = 1.0 / rx; let ry_1 = 1.0 / ry; let rz_1 = 1.0 / rz;

    let dev = CudaDevice::new(0)?;
    let ptx = compile_ptx(KERNEL_SRC)?;
    dev.load_ptx(ptx, "sm", &["hotspot"])?;
    let f = dev.get_func("sm", "hotspot").unwrap();

    let d_p = dev.htod_copy(pvec.clone())?;
    let mut d_a = dev.htod_copy(tvec.clone())?;
    let mut d_b = dev.alloc_zeros::<f32>(l2)?;

    let blk = 16u32;
    let cfg = LaunchConfig {
        grid_dim: (((L as u32) + blk - 1) / blk, ((L as u32) + blk - 1) / blk, 1),
        block_dim: (blk, blk, 1),
        shared_mem_bytes: 0,
    };

    dev.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..ITERS {
        unsafe { f.clone().launch(cfg, (&mut d_b, &d_a, &d_p,
            step_div_cap, rx_1, ry_1, rz_1))?; }
        std::mem::swap(&mut d_a, &mut d_b);
    }
    dev.synchronize()?;
    println!("Total kernel execution time {:.6} (s)", t0.elapsed().as_secs_f64());

    // CPU ref
    let mut cur = tvec.clone(); let mut nxt = vec![0f32; l2];
    for _ in 0..ITERS {
        for y in 0..L { for x in 0..L {
            let n = if y == 0 { 0 } else { y - 1 };
            let sd = if y == L - 1 { L - 1 } else { y + 1 };
            let w = if x == 0 { 0 } else { x - 1 };
            let e = if x == L - 1 { L - 1 } else { x + 1 };
            let idx = y * L + x;
            let t = cur[idx];
            nxt[idx] = t + step_div_cap * (pvec[idx]
                + (cur[sd*L + x] + cur[n*L + x] - 2.0*t) * ry_1
                + (cur[y*L + e] + cur[y*L + w] - 2.0*t) * rx_1
                + (80.0 - t) * rz_1);
        }}
        std::mem::swap(&mut cur, &mut nxt);
    }

    let gpu: Vec<f32> = dev.dtoh_sync_copy(&d_a)?;
    let maxabs = gpu.iter().zip(cur.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    println!("max |err| = {}", maxabs);
    println!("{}", if maxabs < 1e-3 { "PASS" } else { "FAIL" });
    Ok(())
}
