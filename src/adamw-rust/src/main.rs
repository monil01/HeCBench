// adamw-rust: element-wise AdamW optimizer, one thread per parameter.
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::env;
use std::time::Instant;

const KERNEL_SRC: &str = r#"
extern "C" __global__ void adamw(
    float* p, float* m, float* v, const float* g,
    float b1, float b2, float eps, float grad_scale,
    float lr, float decay, int n, int time_step)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    float sg = g[j] / grad_scale;
    float mj = m[j], vj = v[j], pj = p[j];
    for (int t = 1; t <= time_step; t++) {
        mj = b1 * mj + (1.f - b1) * sg;
        vj = b2 * vj + (1.f - b2) * sg * sg;
        float m_hat = mj / (1.f - powf(b1, (float)t));
        float v_hat = vj / (1.f - powf(b2, (float)t));
        pj = pj - lr * (m_hat / (sqrtf(v_hat) + eps) + decay * pj);
    }
    p[j] = pj; m[j] = mj; v[j] = vj;
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let n: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(100_000);
    let time_step: i32 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(10);
    let repeat: i32 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(5);
    let b1: f32 = 0.9; let b2: f32 = 0.999; let eps: f32 = 1e-8;
    let grad_scale: f32 = 1.0; let lr: f32 = 1e-3; let decay: f32 = 1e-2;

    let p_init: Vec<f32> = vec![0.5f32; n];
    let g: Vec<f32> = (0..n).map(|i| ((i % 7) as f32) * 0.01).collect();

    // CPU reference
    let mut pref = p_init.clone();
    let mut mref = vec![0f32; n];
    let mut vref = vec![0f32; n];
    for _ in 0..repeat {
        for j in 0..n {
            let sg = g[j] / grad_scale;
            for t in 1..=time_step {
                mref[j] = b1 * mref[j] + (1. - b1) * sg;
                vref[j] = b2 * vref[j] + (1. - b2) * sg * sg;
                let m_hat = mref[j] / (1. - b1.powi(t));
                let v_hat = vref[j] / (1. - b2.powi(t));
                pref[j] = pref[j] - lr * (m_hat / (v_hat.sqrt() + eps) + decay * pref[j]);
            }
        }
    }

    let dev = CudaDevice::new(0)?;
    let ptx = compile_ptx(KERNEL_SRC)?;
    dev.load_ptx(ptx, "sm", &["adamw"])?;
    let f = dev.get_func("sm", "adamw").unwrap();

    let mut d_p = dev.htod_copy(p_init.clone())?;
    let mut d_m = dev.alloc_zeros::<f32>(n)?;
    let mut d_v = dev.alloc_zeros::<f32>(n)?;
    let d_g = dev.htod_copy(g.clone())?;

    let block = 256u32;
    let grid = ((n as u32) + block - 1) / block;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };

    dev.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..repeat {
        unsafe { f.clone().launch(cfg,
            (&mut d_p, &mut d_m, &mut d_v, &d_g,
             b1, b2, eps, grad_scale, lr, decay, n as i32, time_step))?; }
    }
    dev.synchronize()?;
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / repeat as f64;
    println!("Average kernel execution time: {:.3} (ms)", ms);

    let pgpu: Vec<f32> = dev.dtoh_sync_copy(&d_p)?;
    let maxabs = pgpu.iter().zip(pref.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("max |err| = {}", maxabs);
    println!("{}", if maxabs < 1e-3 { "PASS" } else { "FAIL" });
    Ok(())
}
