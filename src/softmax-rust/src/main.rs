// Softmax port from softmax-cuda. Implements the naive per-slice kernel.
// Optimised warp variant is omitted; the naive kernel is what the CPU
// reference is verified against here (equivalent output modulo FP order).
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::env;
use std::time::Instant;

const KERNEL_SRC: &str = r#"
extern "C" __global__ void softMax(int numSlice, int sliceSize,
                                   const float* __restrict__ src,
                                   float* __restrict__ dest)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numSlice) return;
    float m = src[i * sliceSize];
    for (int j = 0; j < sliceSize; j++) {
        float v = src[i * sliceSize + j];
        if (v > m) m = v;
    }
    float sum = 0.f;
    for (int j = 0; j < sliceSize; j++) {
        sum += expf(src[i * sliceSize + j] - m);
    }
    for (int j = 0; j < sliceSize; j++) {
        dest[i * sliceSize + j] = expf(src[i * sliceSize + j] - m) / sum;
    }
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let num_slice: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(1000);
    let slice_size: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(784);
    let _kernel: i32 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(0);
    let repeat: i32 = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(10);

    let num_elem = num_slice * slice_size;
    // Same rand() % 13 pattern as CUDA main, using seed 2 via a tiny LCG.
    // libc rand seeded with 2 is not portable to Rust; we use a
    // deterministic LCG that also yields ints in [0,13).
    let mut st: u64 = 2;
    let input: Vec<f32> = (0..num_elem).map(|_| {
        st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((st >> 33) % 13) as f32
    }).collect();

    let dev = CudaDevice::new(0)?;
    let ptx = compile_ptx(KERNEL_SRC)?;
    dev.load_ptx(ptx, "sm", &["softMax"])?;
    let f = dev.get_func("sm", "softMax").unwrap();

    let d_in = dev.htod_copy(input.clone())?;
    let mut d_out = dev.alloc_zeros::<f32>(num_elem)?;

    let block = 256u32;
    let grid = ((num_slice as u32) + block - 1) / block;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };

    dev.synchronize()?;
    let start = Instant::now();
    for _ in 0..repeat {
        unsafe { f.clone().launch(cfg, (num_slice as i32, slice_size as i32, &d_in, &mut d_out))?; }
    }
    dev.synchronize()?;
    let elapsed = start.elapsed();
    println!("Average kernel execution time: {:.3} (ms)",
             elapsed.as_secs_f64() * 1000.0 / repeat as f64);

    let gpu: Vec<f32> = dev.dtoh_sync_copy(&d_out)?;

    // CPU reference (mirror of the CUDA softMax_cpu function)
    let mut ok = true;
    for i in 0..num_slice {
        let base = i * slice_size;
        let mut m = input[base];
        for j in 0..slice_size {
            if input[base + j] > m { m = input[base + j]; }
        }
        let mut sum = 0f32;
        let mut e = vec![0f32; slice_size];
        for j in 0..slice_size {
            e[j] = (input[base + j] - m).exp();
            sum += e[j];
        }
        for j in 0..slice_size {
            let cpu = e[j] / sum;
            if (cpu - gpu[base + j]).abs() > 1e-3 {
                println!("@ ({},{}) cpu={} gpu={}", i, j, cpu, gpu[base + j]);
                ok = false;
                break;
            }
        }
        if !ok { break; }
    }
    println!("{}", if ok { "PASS" } else { "FAIL" });
    Ok(())
}
