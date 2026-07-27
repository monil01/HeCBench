// Triad benchmark: C = A + s*B
// Simplified from the SHOC Triad benchmark: one buffer size, one pass loop.
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::env;
use std::time::Instant;

const KERNEL_SRC: &str = r#"
extern "C" __global__ void triad(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 float s,
                                 int n)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid < n) C[gid] = A[gid] + s*B[gid];
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let mut passes = 100i32;
    let mut verbose = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--passes" => { passes = args[i+1].parse()?; i += 2; }
            "-v" | "--verbose" => { verbose = true; i += 1; }
            _ => { i += 1; }
        }
    }

    // Vector length: 4M floats (~16 MB)
    let n: usize = 4 * 1024 * 1024;
    let scalar = 1.75f32;

    let mut rng = StdRng::seed_from_u64(8650341);
    let a: Vec<f32> = (0..n).map(|_| rng.gen_range(0.0..10.0) as f32).collect();
    let b = a.clone();

    let dev = CudaDevice::new(0)?;
    let ptx = compile_ptx(KERNEL_SRC)?;
    dev.load_ptx(ptx, "triad", &["triad"])?;
    let f = dev.get_func("triad", "triad").unwrap();

    let d_a = dev.htod_copy(a.clone())?;
    let d_b = dev.htod_copy(b.clone())?;
    let mut d_c = dev.alloc_zeros::<f32>(n)?;

    let block = 128u32;
    let grid = ((n as u32) + block - 1) / block;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };

    dev.synchronize()?;
    let start = Instant::now();
    for _ in 0..passes {
        unsafe {
            f.clone().launch(cfg, (&d_a, &d_b, &mut d_c, scalar, n as i32))?;
        }
    }
    dev.synchronize()?;
    let elapsed = start.elapsed();

    let c: Vec<f32> = dev.dtoh_sync_copy(&d_c)?;
    let gflops = (n as f64) * 2.0 * (passes as f64) / (elapsed.as_secs_f64() * 1e9);
    let bw = (n as f64) * 4.0 * 3.0 * (passes as f64) / (elapsed.as_secs_f64() * 1e9);

    if verbose {
        println!("N = {}, passes = {}", n, passes);
        println!("Average TriadFlops {:.3} GFLOPS/s", gflops);
        println!("Average TriadBdwth {:.3} GB/s", bw);
    }

    // Verify
    let mut ok = true;
    for j in 0..n {
        let expected = a[j] + scalar * b[j];
        if (c[j] - expected).abs() > 1e-4_f32 * expected.abs().max(1.0) {
            println!("Mismatch at {}: got {}, expected {}", j, c[j], expected);
            ok = false;
            break;
        }
    }
    println!("{}", if ok { "PASS" } else { "FAIL" });
    Ok(())
}
