// Rust port of the `bitonic-sort` HeCBench benchmark.
//
// Array size = 2^n. The GPU sort is a log^2(n)-launch bitonic network: for
// step = 0..n-1 and stage = step..0 the kernel swaps within bitonic
// subsequences of length seq_len = 2^(stage+1). Kernel body is lifted from
// the CUDA source verbatim. Verified against Rust's stable sort.
//
// Usage: bitonic-sort-rust <log2_size> <seed>
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::env;
use std::time::Instant;

const BLOCK_SIZE: u32 = 256;

const KERNEL_SRC: &str = r#"
extern "C" __global__ void
bitonic_sort(const int seq_len, const int two_power, int* a)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int seq_num = i / seq_len;
    int swapped_ele = -1;
    int h_len = seq_len / 2;
    if (i < (seq_len * seq_num) + h_len) swapped_ele = i + h_len;
    int odd = seq_num / two_power;
    bool increasing = ((odd % 2) == 0);
    if (swapped_ele != -1) {
        int ai = a[i];
        int aj = a[swapped_ele];
        if ((ai > aj && increasing) || (ai < aj && !increasing)) {
            a[i] = aj;
            a[swapped_ele] = ai;
        }
    }
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <log2_size> <seed>", args[0]);
        std::process::exit(1);
    }
    let n: u32 = args[1].parse()?;
    let seed: u64 = args[2].parse()?;
    let size: usize = 1usize << n;
    println!("Array size: {}, seed: {}", size, seed);

    // Random input, deterministic
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut input: Vec<i32> = (0..size).map(|_| rng.r#gen::<i32>()).collect();

    // CPU golden
    let mut expected = input.clone();
    expected.sort();

    let dev = CudaDevice::new(0)?;
    let ptx = compile_ptx(KERNEL_SRC)?;
    dev.load_ptx(ptx, "bitonic", &["bitonic_sort"])?;
    let f = dev.get_func("bitonic", "bitonic_sort").unwrap();

    let mut d_a = dev.htod_copy(input.clone())?;

    let grid = (size as u32) / BLOCK_SIZE;
    let cfg = LaunchConfig {
        grid_dim: (grid.max(1), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    dev.synchronize()?;
    let t0 = Instant::now();
    // step = 0..n-1, stage = step..0
    for step in 0..n as i32 {
        for stage in (0..=step).rev() {
            let seq_len: i32 = 1 << (stage + 1);
            let two_power: i32 = 1 << (step - stage);
            unsafe { f.clone().launch(cfg, (seq_len, two_power, &mut d_a))?; }
        }
    }
    dev.synchronize()?;
    let elapsed_ms = t0.elapsed().as_nanos() as f64 * 1e-6;
    println!("Total kernel execution time: {:.6} (ms)", elapsed_ms);

    input = dev.dtoh_sync_copy(&d_a)?;

    let mut ok = true;
    for i in 0..size {
        if input[i] != expected[i] {
            println!("Mismatch at {}: gpu={} cpu={}", i, input[i], expected[i]);
            ok = false;
            if i > 4 { break; }
        }
    }
    println!("{}", if ok { "PASS" } else { "FAIL" });
    Ok(())
}
