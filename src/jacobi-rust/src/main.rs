// 2D Jacobi relaxation, ported from jacobi-cuda. Kernel is a simplified
// version without the warp-shuffle reduction — a separate reduction kernel
// computes the L2 error. Same 2048x2048 grid, same sinusoidal boundary
// conditions.
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::time::Instant;

const N: usize = 2048;

const KERNEL_SRC: &str = r#"
extern "C" __global__ void jacobi_step(
    float* __restrict__ f,
    const float* __restrict__ f_old,
    float* __restrict__ err,
    int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= 1 && i <= N-2 && j >= 1 && j <= N-2) {
        float v = 0.25f * (f_old[(i+1) + j*N] + f_old[(i-1) + j*N]
                         + f_old[i + (j+1)*N] + f_old[i + (j-1)*N]);
        float d = v - f_old[i + j*N];
        f[i + j*N] = v;
        atomicAdd(err, d * d);
    }
}
"#;

fn initialize(f: &mut [f32]) {
    let pi = std::f32::consts::PI;
    for j in 0..N {
        for i in 0..N {
            let idx = i + j * N;
            if i == 0 || i == N-1 {
                f[idx] = (j as f32 * 2.0 * pi / (N as f32 - 1.0)).sin();
            } else if j == 0 || j == N-1 {
                f[idx] = (i as f32 * 2.0 * pi / (N as f32 - 1.0)).sin();
            } else {
                f[idx] = 0.0;
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut f = vec![0f32; N * N];
    initialize(&mut f);
    let f_old = f.clone();

    let dev = CudaDevice::new(0)?;
    let ptx = compile_ptx(KERNEL_SRC)?;
    dev.load_ptx(ptx, "jac", &["jacobi_step"])?;
    let jstep = dev.get_func("jac", "jacobi_step").unwrap();

    let mut d_f: CudaSlice<f32> = dev.htod_copy(f)?;
    let mut d_f_old: CudaSlice<f32> = dev.htod_copy(f_old)?;
    let mut d_err: CudaSlice<f32> = dev.alloc_zeros::<f32>(1)?;

    let cfg = LaunchConfig {
        grid_dim: ((N as u32)/16, (N as u32)/16, 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };

    let mut error = f32::MAX;
    let tolerance = 1e-5f32;
    let max_iters = 10000i32;
    let mut num_iters = 0i32;

    dev.synchronize()?;
    let start = Instant::now();

    while error > tolerance && num_iters < max_iters {
        dev.htod_copy_into(vec![0f32; 1], &mut d_err)?;
        unsafe { jstep.clone().launch(cfg, (&mut d_f, &d_f_old, &mut d_err, N as i32))?; }
        std::mem::swap(&mut d_f, &mut d_f_old);
        let e = dev.dtoh_sync_copy(&d_err)?;
        error = (e[0] / (N as f32 * N as f32)).sqrt();
        if num_iters % 1000 == 0 {
            println!("Error after iteration {} = {}", num_iters, error);
        }
        num_iters += 1;
    }

    dev.synchronize()?;
    let elapsed = start.elapsed();
    println!("Average execution time per iteration: {:.6} (s)",
             elapsed.as_secs_f64() / num_iters as f64);

    if error <= tolerance && num_iters < max_iters {
        println!("PASS");
    } else {
        println!("FAIL (iters={}, err={})", num_iters, error);
        std::process::exit(1);
    }
    Ok(())
}
