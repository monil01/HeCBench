use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::env;
use std::time::Instant;

const BLOCK_SIZE: u32 = 256;
const A1: f32 = 1.5; const A2: f32 = 2.0; const A3: f32 = 2.5; const A4: f32 = 3.0; const A5: f32 = 3.5;
const R_MIN: f32 = 0.0; const R_MAX: f32 = 2.0;
const EPS_TOL: f32 = 1.0;

const KERNEL_SRC: &str = r#"
#define a1 1.5f
#define a2 2.f
#define a3 2.5f
#define a4 3.f
#define a5 3.5f
#define R_min 0.f
#define R_max 2.f
#define BLOCK_SIZE 256

extern "C" __global__ void AIDW_Kernel(
    const float* __restrict__ dx,
    const float* __restrict__ dy,
    const float* __restrict__ dz,
    const int dnum,
    const float* __restrict__ ix,
    const float* __restrict__ iy,
          float* __restrict__ iz,
    const int inum,
    const float area,
    const float* __restrict__ avg_dist)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= inum) return;

    float sum = 0.f, dist = 0.f, t = 0.f, z = 0.f, alpha = 1.f;
    float r_obs = avg_dist[tid];
    float r_exp = 0.5f / sqrtf((float)dnum / area);
    float R_S0 = r_obs / r_exp;

    float u_R = 0.f;
    if (R_S0 >= R_min) u_R = 0.5f - 0.5f * cosf(3.1415926f / R_max * (R_S0 - R_min));
    if (R_S0 >= R_max) u_R = 1.f;

    if (u_R >= 0.f && u_R <= 0.1f) alpha = a1;
    if (u_R > 0.1f && u_R <= 0.3f) alpha = a1*(1.f-5.f*(u_R-0.1f)) + a2*5.f*(u_R-0.1f);
    if (u_R > 0.3f && u_R <= 0.5f) alpha = a3*5.f*(u_R-0.3f) + a1*(1.f-5.f*(u_R-0.3f));
    if (u_R > 0.5f && u_R <= 0.7f) alpha = a3*(1.f-5.f*(u_R-0.5f)) + a4*5.f*(u_R-0.5f);
    if (u_R > 0.7f && u_R <= 0.9f) alpha = a5*5.f*(u_R-0.7f) + a4*(1.f-5.f*(u_R-0.7f));
    if (u_R > 0.9f && u_R <= 1.f)  alpha = a5;
    alpha *= 0.5f;

    for (int j = 0; j < dnum; j++) {
        dist = (ix[tid] - dx[j]) * (ix[tid] - dx[j]) + (iy[tid] - dy[j]) * (iy[tid] - dy[j]);
        t = 1.f / powf(dist, alpha); sum += t; z += dz[j] * t;
    }
    iz[tid] = z / sum;
}

extern "C" __global__ void AIDW_Kernel_Tiled(
    const float* __restrict__ dx,
    const float* __restrict__ dy,
    const float* __restrict__ dz,
    const int dnum,
    const float* __restrict__ ix,
    const float* __restrict__ iy,
          float* __restrict__ iz,
    const int inum,
    const float area,
    const float* __restrict__ avg_dist)
{
    __shared__ float sdx[BLOCK_SIZE];
    __shared__ float sdy[BLOCK_SIZE];
    __shared__ float sdz[BLOCK_SIZE];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= inum) return;

    float dist = 0.f, t = 0.f, alpha = 0.f;
    int part = (dnum - 1) / BLOCK_SIZE;
    int m, e;
    float sum_up = 0.f, sum_dn = 0.f;
    float six_s, siy_s;

    float r_obs = avg_dist[tid];
    float r_exp = 0.5f / sqrtf((float)dnum / area);
    float R_S0 = r_obs / r_exp;

    float u_R = 0.f;
    if (R_S0 >= R_min) u_R = 0.5f - 0.5f * cosf(3.1415926f / R_max * (R_S0 - R_min));
    if (R_S0 >= R_max) u_R = 1.f;

    if (u_R >= 0.f && u_R <= 0.1f) alpha = a1;
    if (u_R > 0.1f && u_R <= 0.3f) alpha = a1*(1.f-5.f*(u_R-0.1f)) + a2*5.f*(u_R-0.1f);
    if (u_R > 0.3f && u_R <= 0.5f) alpha = a3*5.f*(u_R-0.3f) + a1*(1.f-5.f*(u_R-0.3f));
    if (u_R > 0.5f && u_R <= 0.7f) alpha = a3*(1.f-5.f*(u_R-0.5f)) + a4*5.f*(u_R-0.5f);
    if (u_R > 0.7f && u_R <= 0.9f) alpha = a5*5.f*(u_R-0.7f) + a4*(1.f-5.f*(u_R-0.7f));
    if (u_R > 0.9f && u_R <= 1.f)  alpha = a5;
    alpha *= 0.5f;

    float six_t = ix[tid];
    float siy_t = iy[tid];
    int lid = threadIdx.x;
    for (m = 0; m <= part; m++) {
        int num_threads = min(BLOCK_SIZE, dnum - BLOCK_SIZE * m);
        if (lid < num_threads) {
            sdx[lid] = dx[lid + BLOCK_SIZE * m];
            sdy[lid] = dy[lid + BLOCK_SIZE * m];
            sdz[lid] = dz[lid + BLOCK_SIZE * m];
        }
        __syncthreads();
        for (e = 0; e < BLOCK_SIZE; e++) {
            six_s = six_t - sdx[e];
            siy_s = siy_t - sdy[e];
            dist = six_s * six_s + siy_s * siy_s;
            t = 1.f / powf(dist, alpha); sum_dn += t; sum_up += t * sdz[e];
        }
        __syncthreads();
    }
    iz[tid] = sum_up / sum_dn;
}
"#;

fn reference(dx: &[f32], dy: &[f32], dz: &[f32], ix: &[f32], iy: &[f32], iz: &mut [f32],
             area: f32, avg_dist: &[f32]) {
    let dnum = dx.len();
    let inum = ix.len();
    for tid in 0..inum {
        let mut sum = 0f32; let mut z = 0f32; let mut alpha: f32;

        let r_obs = avg_dist[tid];
        let r_exp = 1.0 / (2.0 * ((dnum as f32) / area).sqrt());
        let r_s0 = r_obs / r_exp;
        let mut u_r = 0f32;
        if r_s0 >= R_MIN { u_r = 0.5 - 0.5 * (3.1415926 / R_MAX * (r_s0 - R_MIN)).cos(); }
        if r_s0 >= R_MAX { u_r = 1.0; }

        alpha = 0.0;
        if u_r >= 0.0 && u_r <= 0.1 { alpha = A1; }
        if u_r > 0.1 && u_r <= 0.3 { alpha = A1*(1.0-5.0*(u_r-0.1)) + A2*5.0*(u_r-0.1); }
        if u_r > 0.3 && u_r <= 0.5 { alpha = A3*5.0*(u_r-0.3) + A1*(1.0-5.0*(u_r-0.3)); }
        if u_r > 0.5 && u_r <= 0.7 { alpha = A3*(1.0-5.0*(u_r-0.5)) + A4*5.0*(u_r-0.5); }
        if u_r > 0.7 && u_r <= 0.9 { alpha = A5*5.0*(u_r-0.7) + A4*(1.0-5.0*(u_r-0.7)); }
        if u_r > 0.9 && u_r <= 1.0 { alpha = A5; }
        alpha *= 0.5;

        for j in 0..dnum {
            let dist = (ix[tid] - dx[j]).powi(2) + (iy[tid] - dy[j]).powi(2);
            let t = 1.0 / dist.powf(alpha);
            sum += t;
            z += dz[j] * t;
        }
        iz[tid] = z / sum;
    }
}

// Match C rand() output: LCG defined by glibc rand()
// Actually, we don't need to bit-match C. Just use any deterministic RNG and generate
// the same values on CPU reference and GPU inputs (both use the same data).
struct SimpleRng { state: u64 }
impl SimpleRng {
    fn new(seed: u64) -> Self { Self { state: seed } }
    fn next(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.state >> 33) as u32) as f32 / (u32::MAX as f32)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 { eprintln!("Usage: {} <pts_K> <check> <iterations>", args[0]); std::process::exit(1); }
    let numk: usize = args[1].parse()?;
    let check: i32 = args[2].parse()?;
    let iterations: i32 = args[3].parse()?;

    let dnum = numk * 1024;
    let inum = dnum;
    let width = 2000.0f32; let height = 2000.0f32; let area = width * height;

    let mut rng = SimpleRng::new(123);
    let mut dx = vec![0f32; dnum]; let mut dy = vec![0f32; dnum]; let mut dz = vec![0f32; dnum];
    for i in 0..dnum { dx[i] = rng.next() * 1000.0; dy[i] = rng.next() * 1000.0; dz[i] = rng.next() * 1000.0; }
    let mut ix = vec![0f32; inum]; let mut iy = vec![0f32; inum];
    for i in 0..inum { ix[i] = rng.next() * 1000.0; iy[i] = rng.next() * 1000.0; }
    let mut avg_dist = vec![0f32; dnum];
    for i in 0..dnum { avg_dist[i] = rng.next() * 3.0; }

    println!("Size = : {} K", numk);
    println!("dnum = : {}\ninum = : {}", dnum, inum);

    let mut h_iz = vec![0f32; inum];
    if check != 0 {
        println!("Verification enabled");
        reference(&dx, &dy, &dz, &ix, &iy, &mut h_iz, area, &avg_dist);
    } else {
        println!("Verification disabled");
    }

    let dev = CudaDevice::new(0)?;
    let ptx = compile_ptx(KERNEL_SRC)?;
    dev.load_ptx(ptx, "m", &["AIDW_Kernel", "AIDW_Kernel_Tiled"])?;
    let f_plain = dev.get_func("m", "AIDW_Kernel").unwrap();
    let f_tiled = dev.get_func("m", "AIDW_Kernel_Tiled").unwrap();

    let d_dx = dev.htod_copy(dx.clone())?;
    let d_dy = dev.htod_copy(dy.clone())?;
    let d_dz = dev.htod_copy(dz.clone())?;
    let d_avg = dev.htod_copy(avg_dist.clone())?;
    let d_ix = dev.htod_copy(ix.clone())?;
    let d_iy = dev.htod_copy(iy.clone())?;
    let mut d_iz = dev.alloc_zeros::<f32>(inum)?;

    let cfg = LaunchConfig {
        grid_dim: (((inum as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        f_plain.clone().launch(cfg, (
            &d_dx, &d_dy, &d_dz, dnum as i32, &d_ix, &d_iy, &mut d_iz, inum as i32, area, &d_avg
        ))?;
    }
    let iz_gpu: Vec<f32> = dev.dtoh_sync_copy(&d_iz)?;
    if check != 0 {
        let ok = verify(&iz_gpu, &h_iz, EPS_TOL);
        println!("{}", if ok { "PASS" } else { "FAIL" });
    }

    unsafe {
        f_tiled.clone().launch(cfg, (
            &d_dx, &d_dy, &d_dz, dnum as i32, &d_ix, &d_iy, &mut d_iz, inum as i32, area, &d_avg
        ))?;
    }
    let iz_gpu: Vec<f32> = dev.dtoh_sync_copy(&d_iz)?;
    if check != 0 {
        let ok = verify(&iz_gpu, &h_iz, EPS_TOL);
        println!("{}", if ok { "PASS" } else { "FAIL" });
    }

    dev.synchronize()?;
    let start = Instant::now();
    for _ in 0..iterations {
        unsafe {
            f_plain.clone().launch(cfg, (
                &d_dx, &d_dy, &d_dz, dnum as i32, &d_ix, &d_iy, &mut d_iz, inum as i32, area, &d_avg
            ))?;
        }
    }
    dev.synchronize()?;
    let elapsed = start.elapsed();
    println!("Average execution time of AIDW_Kernel       {} (s)", elapsed.as_secs_f64() / iterations as f64);

    let start = Instant::now();
    for _ in 0..iterations {
        unsafe {
            f_tiled.clone().launch(cfg, (
                &d_dx, &d_dy, &d_dz, dnum as i32, &d_ix, &d_iy, &mut d_iz, inum as i32, area, &d_avg
            ))?;
        }
    }
    dev.synchronize()?;
    let elapsed = start.elapsed();
    println!("Average execution time of AIDW_Kernel_Tiled {} (s)", elapsed.as_secs_f64() / iterations as f64);

    Ok(())
}

fn verify(gpu: &[f32], gold: &[f32], eps: f32) -> bool {
    for i in 0..gpu.len() {
        if (gpu[i] - gold[i]).abs() > eps {
            println!("{} {} {}", i, gold[i], gpu[i]);
            return false;
        }
    }
    true
}
