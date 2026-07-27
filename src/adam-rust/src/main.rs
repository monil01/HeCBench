use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::env;
use std::time::Instant;

const KERNEL_SRC: &str = r#"
extern "C" __global__ void adam_kernel(
    float* __restrict__ p,
    float* __restrict__ m,
    float* __restrict__ v,
    const float* __restrict__ g,
    const float b1,
    const float b2,
    const float eps,
    const float grad_scale,
    const float step_size,
    const int time_step,
    const int vector_size,
    const int mode,
    const float decay)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int totThreads = gridDim.x * blockDim.x;
    for (int j = i; j < vector_size; j += totThreads) {
        for (int t = 1; t <= time_step; t++) {
            float scaled_grad = g[j] / grad_scale;
            m[j] = b1 * m[j] + (1.f - b1) * scaled_grad;
            v[j] = b2 * v[j] + (1.f - b2) * scaled_grad * scaled_grad;
            float m_corrected = m[j] / (1.f - powf(b1, (float)t));
            float v_corrected = v[j] / (1.f - powf(b2, (float)t));
            float denom;
            if (mode == 0) denom = sqrtf(v_corrected + eps);
            else           denom = sqrtf(v_corrected) + eps;
            float update = (m_corrected / denom) + (decay * p[j]);
            p[j] -= (step_size * update);
        }
    }
}
"#;

fn adam_reference(
    p: &mut [f32],
    m: &mut [f32],
    v: &mut [f32],
    g: &[f32],
    b1: f32,
    b2: f32,
    eps: f32,
    grad_scale: f32,
    step_size: f32,
    time_step: i32,
    mode: i32,
    decay: f32,
    repeat: i32,
) {
    for _ in 0..repeat {
        for j in 0..p.len() {
            for t in 1..=time_step {
                let scaled_grad = g[j] / grad_scale;
                m[j] = b1 * m[j] + (1.0 - b1) * scaled_grad;
                v[j] = b2 * v[j] + (1.0 - b2) * scaled_grad * scaled_grad;
                let m_corrected = m[j] / (1.0 - b1.powf(t as f32));
                let v_corrected = v[j] / (1.0 - b2.powf(t as f32));
                let denom = if mode == 0 {
                    (v_corrected + eps).sqrt()
                } else {
                    v_corrected.sqrt() + eps
                };
                let update = m_corrected / denom + decay * p[j];
                p[j] -= step_size * update;
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: {} <vector size> <time steps> <repeat>", args[0]);
        std::process::exit(1);
    }
    let vector_size: usize = args[1].parse()?;
    let time_step: i32 = args[2].parse()?;
    let repeat: i32 = args[3].parse()?;

    let mut rng = StdRng::seed_from_u64(19937);
    let m0: Vec<f32> = (0..vector_size).map(|_| rng.r#gen::<f32>()).collect();
    let v0: Vec<f32> = (0..vector_size).map(|_| rng.r#gen::<f32>()).collect();
    let g0: Vec<f32> = (0..vector_size).map(|_| rng.r#gen::<f32>()).collect();
    let p0: Vec<f32> = (0..vector_size).map(|_| rng.r#gen::<f32>()).collect();

    let step_size = 1e-3f32;
    let decay = 0.5f32;
    let beta1 = 0.9f32;
    let beta2 = 0.999f32;
    let eps = 1e-8f32;
    let grad_scale = 256.0f32;
    let mode: i32 = 0;

    // reference computation on CPU
    let mut r = p0.clone();
    let mut rm = m0.clone();
    let mut rv = v0.clone();
    adam_reference(
        &mut r, &mut rm, &mut rv, &g0, beta1, beta2, eps, grad_scale, step_size, time_step,
        mode, decay, repeat,
    );

    // GPU
    let dev = CudaDevice::new(0)?;
    let ptx = compile_ptx(KERNEL_SRC)?;
    dev.load_ptx(ptx, "m", &["adam_kernel"])?;
    let f = dev.get_func("m", "adam_kernel").unwrap();

    let mut d_m = dev.htod_copy(m0.clone())?;
    let mut d_v = dev.htod_copy(v0.clone())?;
    let d_g = dev.htod_copy(g0.clone())?;
    let mut d_p = dev.htod_copy(p0.clone())?;

    let tpb = 256u32;
    let grid = ((vector_size as u32) + tpb - 1) / tpb;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (tpb, 1, 1),
        shared_mem_bytes: 0,
    };

    dev.synchronize()?;
    let start = Instant::now();
    for _ in 0..repeat {
        unsafe {
            f.clone().launch(
                cfg,
                (
                    &mut d_p,
                    &mut d_m,
                    &mut d_v,
                    &d_g,
                    beta1,
                    beta2,
                    eps,
                    grad_scale,
                    step_size,
                    time_step,
                    vector_size as i32,
                    mode,
                    decay,
                ),
            )?;
        }
    }
    dev.synchronize()?;
    let elapsed = start.elapsed();
    println!(
        "Average kernel execution time {} (ms)",
        elapsed.as_secs_f64() * 1e3 / repeat as f64
    );

    let p_out: Vec<f32> = dev.dtoh_sync_copy(&d_p)?;
    let mut ok = true;
    let mut cr = 0.0f64;
    let mut cp = 0.0f64;
    for i in 0..vector_size {
        if (r[i] - p_out[i]).abs() > 1e-3 {
            ok = false;
            break;
        }
        cr += r[i] as f64;
        cp += p_out[i] as f64;
    }
    println!("{}", if ok { "PASS" } else { "FAIL" });
    println!(
        "Checksum: {} {}",
        cr / vector_size as f64,
        cp / vector_size as f64
    );

    Ok(())
}
