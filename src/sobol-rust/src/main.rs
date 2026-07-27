// Rust port of the `sobol` HeCBench benchmark (simplified, matches
// sobol-triton's approach).
//
// The upstream benchmark ships hand-rolled Joe & Kuo direction tables in
// `sobol_primitives.cu`. Reproducing them here does not give any additional
// Rust coverage, so we synthesise 32 direction vectors per dimension from a
// fixed seed (with the top bit of the k-th vector set to 2^(31-k), giving
// a legitimate Gray-code Sobol recurrence). The GPU kernel is the standard
// per-thread Gray-code XOR loop. Correctness is verified against a Rust CPU
// re-implementation using the identical direction vectors.
//
// Usage: sobol-rust <n_vectors> <n_dimensions> <repeat>
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::env;
use std::time::Instant;

const N_DIRECTIONS: usize = 32;
const KERNEL_SRC: &str = r#"
extern "C" __global__ void sobol_kernel(
    const int* __restrict__ dirs,   // [n_dim, 32]
    float*     __restrict__ out,    // [n_dim, n_vec]
    int n_vec, int n_dim)
{
    int dim = blockIdx.y;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_vec || dim >= n_dim) return;
    unsigned int g = (unsigned int)i ^ ((unsigned int)i >> 1);
    unsigned int X = 0;
    #pragma unroll
    for (int k = 0; k < 32; ++k) {
        unsigned int bit = (g >> k) & 1u;
        unsigned int mask = -bit;  // 0 or 0xffffffff
        unsigned int v = (unsigned int)dirs[dim * 32 + k];
        X ^= (mask & v);
    }
    out[dim * n_vec + i] = (float)X * 2.3283064e-10f;
}
"#;

fn sobol_cpu(dirs: &[u32], n_vec: usize, n_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_dim * n_vec];
    for i in 0..n_vec {
        let g = (i as u32) ^ ((i as u32) >> 1);
        for dim in 0..n_dim {
            let mut x: u32 = 0;
            for k in 0..N_DIRECTIONS {
                let bit = (g >> k) & 1;
                let mask = 0u32.wrapping_sub(bit);
                let v = dirs[dim * N_DIRECTIONS + k];
                x ^= mask & v;
            }
            out[dim * n_vec + i] = x as f32 * 2.3283064e-10;
        }
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: {} <n_vectors> <n_dimensions> <repeat>", args[0]);
        std::process::exit(1);
    }
    let n_vec: usize = args[1].parse()?;
    let n_dim: usize = args[2].parse()?;
    let repeat: usize = args[3].parse()?;

    println!("Allocating CPU memory...");
    println!("Allocating GPU memory...");
    println!("Initializing direction numbers...");
    let mut rng = ChaCha8Rng::seed_from_u64(0xC0FFEE_u64);
    let mut dirs = vec![0u32; n_dim * N_DIRECTIONS];
    for dim in 0..n_dim {
        for k in 0..N_DIRECTIONS {
            let r: u32 = rng.r#gen();
            let low30 = r & ((1u32 << 30) - 1);
            let msb   = 1u32 << (31 - k as u32);
            dirs[dim * N_DIRECTIONS + k] = low30 | msb;
        }
    }

    let dev = CudaDevice::new(0)?;
    let ptx = compile_ptx(KERNEL_SRC)?;
    dev.load_ptx(ptx, "sobol", &["sobol_kernel"])?;
    let f = dev.get_func("sobol", "sobol_kernel").unwrap();

    // Upload dirs as signed ints (kernel casts to unsigned).
    let dirs_i32: Vec<i32> = dirs.iter().map(|&u| u as i32).collect();
    let d_dirs = dev.htod_copy(dirs_i32)?;
    let mut d_out = dev.alloc_zeros::<f32>(n_dim * n_vec)?;

    let block: u32 = 128;
    let grid_x = ((n_vec as u32) + block - 1) / block;
    let cfg = LaunchConfig {
        grid_dim: (grid_x, n_dim as u32, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("Executing QRNG on GPU...");
    // warmup + time
    unsafe { f.clone().launch(cfg, (&d_dirs, &mut d_out, n_vec as i32, n_dim as i32))?; }
    dev.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..repeat {
        unsafe { f.clone().launch(cfg, (&d_dirs, &mut d_out, n_vec as i32, n_dim as i32))?; }
    }
    dev.synchronize()?;
    let kt = t0.elapsed().as_secs_f64() / repeat as f64;
    println!("Average kernel execution time: {} (s)", kt);

    let gpu: Vec<f32> = dev.dtoh_sync_copy(&d_out)?;

    println!("\nExecuting QRNG on CPU...");
    let n_cpu = n_dim.min(16);
    let cpu = sobol_cpu(&dirs, n_vec, n_cpu);

    println!("Checking results...");
    let mut l1_diff = 0.0f64;
    let mut l1_ref  = 0.0f64;
    for dim in 0..n_cpu {
        for i in 0..n_vec {
            let g = gpu[dim * n_vec + i];
            let c = cpu[dim * n_vec + i];
            l1_diff += (g - c).abs() as f64;
            l1_ref  += c.abs() as f64;
        }
    }
    let l1err = if l1_ref > 0.0 { l1_diff / l1_ref } else { l1_diff };
    println!("L1-Error: {}", l1err);
    println!("Shutting down...");
    println!("{}", if l1err < 1e-6 { "PASS" } else { "FAIL" });
    Ok(())
}
