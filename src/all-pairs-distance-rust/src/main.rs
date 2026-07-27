// all-pairs-distance-rust: one thread per (gx,gy) pair, atomic-free.
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::env;
use std::time::Instant;

const INSTANCES: usize = 512;
const ATTRIBUTES: usize = 100;

const KERNEL_SRC: &str = r#"
#define INSTANCES 512
#define ATTRIBUTES 100
extern "C" __global__ void apd(const unsigned char* data, int* distance)
{
    int gx = blockIdx.x;
    int gy = blockIdx.y;
    if (gx >= INSTANCES || gy >= INSTANCES) return;
    int cnt = 0;
    for (int i = 0; i < ATTRIBUTES; i++) {
        if (data[i + ATTRIBUTES * gx] != data[i + ATTRIBUTES * gy]) cnt++;
    }
    distance[INSTANCES * gx + gy] = cnt;
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let iters: i32 = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(1);

    let mut s: u64 = 20260721;
    let data: Vec<u8> = (0..(INSTANCES * ATTRIBUTES)).map(|_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) % 4) as u8
    }).collect();

    let dev = CudaDevice::new(0)?;
    let ptx = compile_ptx(KERNEL_SRC)?;
    dev.load_ptx(ptx, "sm", &["apd"])?;
    let f = dev.get_func("sm", "apd").unwrap();

    let d_data = dev.htod_copy(data.clone())?;
    let mut d_dist = dev.alloc_zeros::<i32>(INSTANCES * INSTANCES)?;
    let cfg = LaunchConfig {
        grid_dim: (INSTANCES as u32, INSTANCES as u32, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    dev.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..iters {
        unsafe { f.clone().launch(cfg, (&d_data, &mut d_dist))?; }
    }
    dev.synchronize()?;
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("Average kernel execution time: {:.3} (ms)", ms);

    let mut refv = vec![0i32; INSTANCES * INSTANCES];
    for gx in 0..INSTANCES {
        for gy in 0..INSTANCES {
            let mut cnt = 0i32;
            for i in 0..ATTRIBUTES {
                if data[i + ATTRIBUTES * gx] != data[i + ATTRIBUTES * gy] { cnt += 1; }
            }
            refv[INSTANCES * gx + gy] = cnt;
        }
    }

    let gpu: Vec<i32> = dev.dtoh_sync_copy(&d_dist)?;
    let diff: i64 = gpu.iter().zip(refv.iter())
        .map(|(a, b)| (a - b).abs() as i64).sum();
    println!("diff = {}", diff);
    println!("{}", if diff == 0 { "PASS" } else { "FAIL" });
    Ok(())
}
