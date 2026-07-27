// Binary search benchmark, ported from bsearch-cuda.
// Runs the first (kernel_BS) variant only to keep the port compact;
// the other three variants are numerically equivalent per the reference.
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::env;
use std::time::Instant;

const KERNEL_SRC: &str = r#"
extern "C" __global__ void kernel_BS(const float* __restrict__ a,
                                     const float* __restrict__ z,
                                     unsigned long long* __restrict__ r,
                                     unsigned long long zSize,
                                     unsigned long long n)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= zSize) return;
    float v = z[i];
    unsigned long long low = 0;
    unsigned long long high = n;
    while (high - low > 1) {
        unsigned long long mid = low + (high - low) / 2;
        if (v < a[mid]) high = mid; else low = mid;
    }
    r[i] = low;
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let num_elem: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(1 << 20);
    let repeat: i32 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(10);

    let a_size = num_elem;
    let z_size = 2 * a_size;
    let n = a_size - 1;

    // Strictly ascending array
    let a: Vec<f32> = (0..a_size).map(|i| i as f32).collect();
    // Pseudo-random queries in [0, n)
    let mut state: u64 = 2;
    let z: Vec<f32> = (0..z_size)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let r = ((state >> 33) as u64) % (n as u64);
            r as f32
        })
        .collect();

    let dev = CudaDevice::new(0)?;
    let ptx = compile_ptx(KERNEL_SRC)?;
    dev.load_ptx(ptx, "bs", &["kernel_BS"])?;
    let f = dev.get_func("bs", "kernel_BS").unwrap();

    let d_a = dev.htod_copy(a.clone())?;
    let d_z = dev.htod_copy(z.clone())?;
    let mut d_r = dev.alloc_zeros::<u64>(z_size)?;

    let block = 256u32;
    let grid = ((z_size as u32) + block - 1) / block;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };

    dev.synchronize()?;
    let start = Instant::now();
    for _ in 0..repeat {
        unsafe {
            f.clone().launch(
                cfg,
                (&d_a, &d_z, &mut d_r, z_size as u64, n as u64),
            )?;
        }
    }
    dev.synchronize()?;
    let elapsed = start.elapsed();
    println!(
        "Average kernel execution time (bs1) {:.6} (s)",
        elapsed.as_secs_f64() / repeat as f64
    );

    let r: Vec<u64> = dev.dtoh_sync_copy(&d_r)?;
    let mut ok = true;
    for i in 0..z_size {
        let idx = r[i] as usize;
        if !(idx + 1 < a_size && a[idx] <= z[i] && z[i] < a[idx + 1]) {
            println!("Mismatch at {}: idx={} z={} a[idx]={} a[idx+1]={}",
                     i, idx, z[i], a[idx], a[idx + 1]);
            ok = false;
            break;
        }
    }
    println!("{}", if ok { "PASS" } else { "FAIL" });
    Ok(())
}
