// saxpy-ompt-rust: element-wise y = a*x + y, N launches for timing.
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::env;
use std::time::Instant;

const KERNEL_SRC: &str = r#"
extern "C" __global__ void saxpy(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let n: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(1 << 22);
    let repeat: i32 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(100);
    let a: f32 = 2.0;

    let x: Vec<f32> = (0..n).map(|i|
        1.0 + ((i as u32).wrapping_mul(2654435761) & 0xffff) as f32 / 65535.0).collect();
    let y_init: Vec<f32> = (0..n).map(|i|
        0.5 + ((i as u32).wrapping_mul(40503) & 0xffff) as f32 / 65535.0).collect();
    let y_expected: Vec<f32> = y_init.iter().zip(x.iter())
        .map(|(y0, xv)| y0 + repeat as f32 * a * xv).collect();

    let dev = CudaDevice::new(0)?;
    let ptx = compile_ptx(KERNEL_SRC)?;
    dev.load_ptx(ptx, "sm", &["saxpy"])?;
    let f = dev.get_func("sm", "saxpy").unwrap();

    let d_x = dev.htod_copy(x.clone())?;
    let mut d_y = dev.htod_copy(y_init.clone())?;

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
        unsafe { f.clone().launch(cfg, (n as i32, a, &d_x, &mut d_y))?; }
    }
    dev.synchronize()?;
    let elapsed = t0.elapsed();
    println!("Average kernel execution time: {:.3} (ms)",
             elapsed.as_secs_f64() * 1000.0 / repeat as f64);

    let ygpu: Vec<f32> = dev.dtoh_sync_copy(&d_y)?;
    let maxabs = ygpu.iter().zip(y_expected.iter())
        .map(|(g, e)| (g - e).abs())
        .fold(0f32, f32::max);
    println!("max |err| = {}", maxabs);
    println!("{}", if maxabs < 1e-2 { "PASS" } else { "FAIL" });
    Ok(())
}
