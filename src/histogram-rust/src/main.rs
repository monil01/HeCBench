// Histogram port from histogram-cuda. Simplified to a single-channel 256-bin
// histogram over a random synthetic uchar image, using the two-kernel
// privatised-partial-histograms pattern from histogram_gmem_atomics.h
// (NUM_PARTS=256, ACTIVE_CHANNELS=1, NUM_BINS=256).
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::env;
use std::time::Instant;

const KERNEL_SRC: &str = r#"
extern "C" __global__ void histogram_gmem_atomics(
    const unsigned char* __restrict__ in,
    int width, int height,
    unsigned int* __restrict__ out)
{
    const int NUM_PARTS = 256;
    const int NUM_BINS  = 256;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int nx = blockDim.x * gridDim.x;
    int ny = blockDim.y * gridDim.y;
    int t = threadIdx.x + threadIdx.y * blockDim.x;
    int nt = blockDim.x * blockDim.y;
    int g = blockIdx.x + blockIdx.y * gridDim.x;
    unsigned int *gmem = out + g * NUM_PARTS;
    for (int i = t; i < NUM_BINS; i += nt) gmem[i] = 0;
    __syncthreads();
    for (int col = x; col < width; col += nx) {
        for (int row = y; row < height; row += ny) {
            unsigned int bin = in[row * width + col];
            atomicAdd(&gmem[bin], 1);
        }
    }
}

extern "C" __global__ void histogram_gmem_accum(
    const unsigned int* __restrict__ in,
    int n,
    unsigned int* __restrict__ out)
{
    const int NUM_PARTS = 256;
    const int NUM_BINS  = 256;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NUM_BINS) return;
    unsigned int total = 0;
    for (int j = 0; j < n; j++) total += in[i + NUM_PARTS * j];
    out[i] = total;
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let repeat: i32 = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(100);

    let width = 1024usize;
    let height = 1024usize;
    let n = width * height;
    let mut rng = StdRng::seed_from_u64(0xB1AB1A);
    let img: Vec<u8> = (0..n).map(|_| rng.gen_range(0..=255u8)).collect();

    let dev = CudaDevice::new(0)?;
    let ptx = compile_ptx(KERNEL_SRC)?;
    dev.load_ptx(ptx, "hist", &["histogram_gmem_atomics", "histogram_gmem_accum"])?;
    let f1 = dev.get_func("hist", "histogram_gmem_atomics").unwrap();
    let f2 = dev.get_func("hist", "histogram_gmem_accum").unwrap();

    let d_in = dev.htod_copy(img.clone())?;
    let total_blocks = 16u32 * 16u32; // grid(16,16)
    let mut d_part = dev.alloc_zeros::<u32>((total_blocks as usize) * 256)?;
    let mut d_hist = dev.alloc_zeros::<u32>(256)?;

    let cfg1 = LaunchConfig {
        grid_dim: (16, 16, 1),
        block_dim: (32, 4, 1),
        shared_mem_bytes: 0,
    };
    let cfg2 = LaunchConfig {
        grid_dim: ((256 + 127) / 128, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };

    dev.synchronize()?;
    let start = Instant::now();
    for _ in 0..repeat {
        unsafe { f1.clone().launch(cfg1, (&d_in, width as i32, height as i32, &mut d_part))?; }
        unsafe { f2.clone().launch(cfg2, (&d_part, total_blocks as i32, &mut d_hist))?; }
    }
    dev.synchronize()?;
    let elapsed = start.elapsed();
    println!("Average kernel time: {:.3} (us)",
             elapsed.as_micros() as f64 / repeat as f64);

    let gpu_hist: Vec<u32> = dev.dtoh_sync_copy(&d_hist)?;
    let mut ref_hist = [0u32; 256];
    for &b in &img { ref_hist[b as usize] += 1; }

    let mut ok = true;
    let mut total_gpu: u64 = 0;
    for i in 0..256 {
        total_gpu += gpu_hist[i] as u64;
        if gpu_hist[i] != ref_hist[i] {
            println!("Mismatch bin {}: gpu={} ref={}", i, gpu_hist[i], ref_hist[i]);
            ok = false;
        }
    }
    println!("Total pixels binned = {} (expected {})", total_gpu, n);
    println!("{}", if ok { "PASS" } else { "FAIL" });
    Ok(())
}
