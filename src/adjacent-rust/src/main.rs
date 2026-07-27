use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::env;
use std::time::Instant;

// ITEMS_PER_THREAD = 4
const KERNEL_SRC: &str = r#"
extern "C" __global__ void adj_diff(const int* __restrict__ d_in, int* __restrict__ d_out,
                                    int items_per_block, int subtract_left) {
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int base = blockIdx.x * items_per_block;

    int t0 = d_in[base + tid*4 + 0];
    int t1 = d_in[base + tid*4 + 1];
    int t2 = d_in[base + tid*4 + 2];
    int t3 = d_in[base + tid*4 + 3];
    sdata[tid*4 + 0] = t0;
    sdata[tid*4 + 1] = t1;
    sdata[tid*4 + 2] = t2;
    sdata[tid*4 + 3] = t3;
    __syncthreads();

    int o0, o1, o2, o3;
    if (subtract_left) {
        int left = (tid == 0) ? 0 : sdata[tid*4 - 1];
        o0 = (tid == 0) ? t0 : (t0 - left);
        o1 = t1 - t0;
        o2 = t2 - t1;
        o3 = t3 - t2;
    } else {
        int last_idx = tid*4 + 4;
        int right = (last_idx < items_per_block) ? sdata[last_idx] : 0;
        o0 = t0 - t1;
        o1 = t1 - t2;
        o2 = t2 - t3;
        o3 = (last_idx >= items_per_block) ? t3 : (t3 - right);
    }
    d_out[base + tid*4 + 0] = o0;
    d_out[base + tid*4 + 1] = o1;
    d_out[base + tid*4 + 2] = o2;
    d_out[base + tid*4 + 3] = o3;
}
"#;

fn test_block(dev: &std::sync::Arc<CudaDevice>, block_threads: u32, num_items_in: usize, repeat: i32)
    -> Result<(), Box<dyn std::error::Error>> {
    let items_per_thread = 4usize;
    let items_per_block = block_threads as usize * items_per_thread;
    let num_items = ((num_items_in + items_per_block - 1) / items_per_block) * items_per_block;

    let mut h_in = vec![0i32; num_items];
    for i in 0..num_items { h_in[i] = (i as i32) % 17; }

    let grid_size = num_items / items_per_block;
    let cfg = LaunchConfig {
        grid_dim: (grid_size as u32, 1, 1),
        block_dim: (block_threads, 1, 1),
        shared_mem_bytes: (items_per_block * std::mem::size_of::<i32>()) as u32,
    };

    let d_in = dev.htod_copy(h_in.clone())?;
    let mut d_out = dev.alloc_zeros::<i32>(num_items)?;
    let f = dev.get_func("m", "adj_diff").unwrap();

    // SubtractLeft
    dev.htod_copy_into(h_in.clone(), &mut d_out.clone())?;
    for _ in 0..repeat {
        unsafe { f.clone().launch(cfg, (&d_in, &mut d_out, items_per_block as i32, 1i32))?; }
    }
    let h_out: Vec<i32> = dev.dtoh_sync_copy(&d_out)?;
    // Reference for SubtractLeft
    let mut r_out = vec![0i32; num_items];
    for b in 0..grid_size {
        for i in 0..items_per_block {
            let idx = b*items_per_block + i;
            r_out[idx] = if i == 0 { h_in[idx] } else { h_in[idx] - h_in[idx-1] };
        }
    }
    let pass_left = r_out == h_out;
    println!("SubtractLeft (block_threads={}): {}", block_threads, if pass_left { "PASS" } else { "FAIL" });

    // SubtractRight
    for _ in 0..repeat {
        unsafe { f.clone().launch(cfg, (&d_in, &mut d_out, items_per_block as i32, 0i32))?; }
    }
    let h_out: Vec<i32> = dev.dtoh_sync_copy(&d_out)?;
    for b in 0..grid_size {
        for i in 0..items_per_block {
            let idx = b*items_per_block + i;
            r_out[idx] = if i+1 >= items_per_block { h_in[idx] } else { h_in[idx] - h_in[idx+1] };
        }
    }
    let pass_right = r_out == h_out;
    println!("SubtractRight (block_threads={}): {}", block_threads, if pass_right { "PASS" } else { "FAIL" });

    // Timing (SubtractLeft then SubtractRight)
    let mut d_tmp = dev.alloc_zeros::<i32>(num_items)?;
    dev.synchronize()?;
    let start = Instant::now();
    for _ in 0..repeat {
        unsafe {
            f.clone().launch(cfg, (&d_in, &mut d_out, items_per_block as i32, 1i32))?;
            f.clone().launch(cfg, (&d_out, &mut d_tmp, items_per_block as i32, 0i32))?;
        }
    }
    dev.synchronize()?;
    let elapsed = start.elapsed();
    println!(
        "Average execution time of the kernels (thread block size = {:4}): {} (us)",
        block_threads, elapsed.as_secs_f64() * 1e6 / repeat as f64
    );

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <number of elements> <repeat>", args[0]);
        std::process::exit(1);
    }
    let nelems: usize = args[1].parse()?;
    let repeat: i32 = args[2].parse()?;

    let dev = CudaDevice::new(0)?;
    let ptx = compile_ptx(KERNEL_SRC)?;
    dev.load_ptx(ptx, "m", &["adj_diff"])?;

    for &bs in &[64u32, 128, 256, 512, 1024] {
        test_block(&dev, bs, nelems, repeat)?;
    }
    Ok(())
}
