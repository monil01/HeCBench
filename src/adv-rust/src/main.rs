// adv-rust: 1D advection surrogate, periodic BC, N iters double-buffered.
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::env;
use std::time::Instant;

const KERNEL_SRC: &str = r#"
extern "C" __global__ void adv(int n,
                               const float* __restrict__ u,
                               const float* __restrict__ g,
                                     float* __restrict__ un)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int im = (i - 1 + n) % n;
    int ip = (i + 1) % n;
    un[i] = u[i] + g[i] * (u[im] - u[ip]);
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let iters: i32 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(100);
    let n: usize = 65536;

    let mut s: u64 = 20260721;
    let mut u = vec![0f32; n];
    let mut g = vec![0f32; n];
    for i in 0..n {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        u[i] = (((s >> 33) & 0x7fffffff) as f32) / 0x7fffffff as f32;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        g[i] = (((s >> 33) & 0x7fffffff) as f32) / 0x7fffffff as f32 * 0.01;
    }

    let dev = CudaDevice::new(0)?;
    let ptx = compile_ptx(KERNEL_SRC)?;
    dev.load_ptx(ptx, "sm", &["adv"])?;
    let f = dev.get_func("sm", "adv").unwrap();

    let mut d_u = dev.htod_copy(u.clone())?;
    let d_g = dev.htod_copy(g.clone())?;
    let mut d_n = dev.alloc_zeros::<f32>(n)?;

    let block = 256u32;
    let grid = ((n as u32) + block - 1) / block;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };

    dev.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..iters {
        unsafe { f.clone().launch(cfg, (n as i32, &d_u, &d_g, &mut d_n))?; }
        std::mem::swap(&mut d_u, &mut d_n);
    }
    dev.synchronize()?;
    let us_per = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    println!("elapsed time= {:.3} us/iter", us_per);

    // CPU reference
    let mut uref = u.clone();
    let mut tmp = vec![0f32; n];
    for _ in 0..iters {
        for i in 0..n {
            let im = (i + n - 1) % n;
            let ip = (i + 1) % n;
            tmp[i] = uref[i] + g[i] * (uref[im] - uref[ip]);
        }
        std::mem::swap(&mut uref, &mut tmp);
    }

    let gpu: Vec<f32> = dev.dtoh_sync_copy(&d_u)?;
    let maxabs = gpu.iter().zip(uref.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("Max error: {}", maxabs);
    println!("{}", if maxabs <= 1e-3 { "PASS" } else { "FAIL" });
    Ok(())
}
