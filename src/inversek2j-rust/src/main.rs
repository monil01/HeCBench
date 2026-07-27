// Rust port of the `inversek2j` HeCBench benchmark.
//
// 3-joint planar inverse kinematics via cyclic coordinate descent. The CUDA
// kernel is compiled at runtime via NVRTC; host code + CPU reference are
// idiomatic Rust. Verifies against a byte-for-byte CPU replay of the same
// algorithm and prints PASS if the number of angle mismatches (|Δ|>1e-3) is
// zero.
//
// Usage: inversek2j-rust <coord_in.txt> <iterations>
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

const NUM_JOINTS: usize = 3;
const BLOCK_SIZE: u32 = 128;
const MAX_LOOP: usize = 25;

const KERNEL_SRC: &str = r#"
#define MAX_LOOP 25
#define NUM_JOINTS 3
#define NUM_JOINTS_P1 4
#define PI 3.14159265358979f

extern "C" __global__ void
invkin_kernel(const float* __restrict__ xTarget_in,
              const float* __restrict__ yTarget_in,
              float* __restrict__ angles,
              int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float angle_out[NUM_JOINTS];
    float curr_x = xTarget_in[idx];
    float curr_y = yTarget_in[idx];
    for (int i = 0; i < NUM_JOINTS; i++) angle_out[i] = 0.0f;

    float xData[NUM_JOINTS_P1];
    float yData[NUM_JOINTS_P1];
    for (int i = 0; i < NUM_JOINTS_P1; i++) { xData[i] = (float)i; yData[i] = 0.f; }

    for (int curr_loop = 0; curr_loop < MAX_LOOP; curr_loop++) {
        for (int iter = NUM_JOINTS; iter > 0; iter--) {
            float pe_x = xData[NUM_JOINTS];
            float pe_y = yData[NUM_JOINTS];
            float pc_x = xData[iter - 1];
            float pc_y = yData[iter - 1];
            float dx1 = pe_x - pc_x, dy1 = pe_y - pc_y;
            float dx2 = curr_x - pc_x, dy2 = curr_y - pc_y;
            float l1 = sqrtf(dx1*dx1 + dy1*dy1);
            float l2 = sqrtf(dx2*dx2 + dy2*dy2);
            float ax = dx1 / l1, ay = dy1 / l1;
            float bx = dx2 / l2, by = dy2 / l2;
            float dot = ax*bx + ay*by;
            if (dot > 1.f) dot = 1.f;
            else if (dot < -1.f) dot = -1.f;
            float ang = acosf(dot) * (180.f / PI);
            float direction = ax*by - ay*bx;
            if (direction < 0.f) ang = -ang;
            if (ang > 30.f) ang = 30.f;
            else if (ang < -30.f) ang = -30.f;
            angle_out[iter - 1] = ang;
            for (int i = 0; i < NUM_JOINTS; i++) {
                if (i < NUM_JOINTS - 1)
                    angle_out[i + 1] += angle_out[i];
            }
        }
    }

    angles[idx * NUM_JOINTS + 0] = angle_out[0];
    angles[idx * NUM_JOINTS + 1] = angle_out[1];
    angles[idx * NUM_JOINTS + 2] = angle_out[2];
}
"#;

/// CPU reference — byte-for-byte port of the CUDA kernel body.
fn invkin_cpu(xs: &[f32], ys: &[f32]) -> Vec<f32> {
    let n = xs.len();
    let mut out = vec![0.0f32; n * NUM_JOINTS];
    for idx in 0..n {
        let curr_x = xs[idx];
        let curr_y = ys[idx];
        let mut angle_out = [0.0f32; NUM_JOINTS];
        let mut x_data = [0.0f32; NUM_JOINTS + 1];
        let mut y_data = [0.0f32; NUM_JOINTS + 1];
        for i in 0..NUM_JOINTS + 1 {
            x_data[i] = i as f32;
            y_data[i] = 0.0;
        }
        for _ in 0..MAX_LOOP {
            for iter in (1..=NUM_JOINTS).rev() {
                let pe_x = x_data[NUM_JOINTS];
                let pe_y = y_data[NUM_JOINTS];
                let pc_x = x_data[iter - 1];
                let pc_y = y_data[iter - 1];
                let dx1 = pe_x - pc_x; let dy1 = pe_y - pc_y;
                let dx2 = curr_x - pc_x; let dy2 = curr_y - pc_y;
                let l1 = (dx1*dx1 + dy1*dy1).sqrt();
                let l2 = (dx2*dx2 + dy2*dy2).sqrt();
                let ax = dx1 / l1; let ay = dy1 / l1;
                let bx = dx2 / l2; let by = dy2 / l2;
                let mut dot = ax*bx + ay*by;
                if dot > 1.0 { dot = 1.0; } else if dot < -1.0 { dot = -1.0; }
                let mut ang = dot.acos() * (180.0 / std::f32::consts::PI);
                let direction = ax*by - ay*bx;
                if direction < 0.0 { ang = -ang; }
                if ang > 30.0 { ang = 30.0; } else if ang < -30.0 { ang = -30.0; }
                angle_out[iter - 1] = ang;
                for i in 0..NUM_JOINTS - 1 {
                    angle_out[i + 1] += angle_out[i];
                }
            }
        }
        out[idx * NUM_JOINTS + 0] = angle_out[0];
        out[idx * NUM_JOINTS + 1] = angle_out[1];
        out[idx * NUM_JOINTS + 2] = angle_out[2];
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <coord_in.txt> <iterations>", args[0]);
        std::process::exit(1);
    }
    let filename = &args[1];
    let iterations: usize = args[2].parse()?;

    let file = File::open(filename)?;
    let mut lines = BufReader::new(file).lines();
    let n: usize = lines.next().unwrap()?.trim().parse()?;
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let line = lines.next().unwrap()?;
        let mut it = line.split_whitespace();
        xs.push(it.next().unwrap().parse::<f32>()?);
        ys.push(it.next().unwrap().parse::<f32>()?);
    }
    println!("# Data Size = {}", n);

    let dev = CudaDevice::new(0)?;
    let ptx = compile_ptx(KERNEL_SRC)?;
    dev.load_ptx(ptx, "invkin", &["invkin_kernel"])?;
    let f = dev.get_func("invkin", "invkin_kernel").unwrap();

    let d_x = dev.htod_copy(xs.clone())?;
    let d_y = dev.htod_copy(ys.clone())?;
    let mut d_a = dev.alloc_zeros::<f32>(n * NUM_JOINTS)?;

    let grid = ((n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (BLOCK_SIZE, 1, 1), shared_mem_bytes: 0 };

    dev.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..iterations {
        unsafe { f.clone().launch(cfg, (&d_x, &d_y, &mut d_a, n as i32))?; }
    }
    dev.synchronize()?;
    let elapsed_us = t0.elapsed().as_nanos() as f64 * 1e-3 / iterations as f64;
    println!("Average kernel execution time {} (us)", elapsed_us);

    let gpu: Vec<f32> = dev.dtoh_sync_copy(&d_a)?;
    let cpu = invkin_cpu(&xs, &ys);

    let mut errors = 0usize;
    for i in 0..(n * NUM_JOINTS) {
        if (gpu[i] - cpu[i]).abs() > 1e-3 { errors += 1; }
    }
    println!("{}", if errors == 0 { "PASS" } else { "FAIL" });
    Ok(())
}
