// daphne-rust: LiDAR-to-image projection surrogate mirroring the Triton
// reference at ../daphne-triton/main.py. Synthetic LCG point cloud;
// per-point extrinsic rotation, radial+tangential undistortion, intrinsic
// projection, pt2>2.5 visibility filter. Verified against a host reference.

use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::env;
use std::time::Instant;

const KERNEL_SRC: &str = r#"
#define R00 -0.9111348390579224f
#define R01  0.0751304179430008f
#define R02 -0.4052018225193024f
#define R10 -0.3360927104949951f
#define R11 -0.7044632434844971f
#define R12  0.6251187324523926f
#define R20 -0.2384843230247498f
#define R21  0.7057529091835022f
#define R22  0.6671117544174194f
#define T0 0.1f
#define T1 -0.2f
#define T2 0.3f
#define D0 0.03f
#define D1 -0.15f
#define D2 0.001f
#define D3 0.001f
#define D4 0.05f
#define FX 1200.0f
#define CX 400.0f
#define FY 1200.0f
#define CY 300.0f
#define POINT_STEP 8

extern "C" __global__ void project(const float* __restrict__ cp,
                                   float* __restrict__ ox,
                                   float* __restrict__ oy,
                                   float* __restrict__ oz,
                                   int*   __restrict__ ov,
                                   int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* pt = cp + i * POINT_STEP;
    float p0 = pt[0], p1 = pt[1], p2 = pt[2];
    float pt0 = T0 + p0 * R00 + p1 * R01 + p2 * R02;
    float pt1 = T1 + p0 * R10 + p1 * R11 + p2 * R12;
    float pt2 = T2 + p0 * R20 + p1 * R21 + p2 * R22;
    bool close = pt2 > 2.5f;
    float denom = close ? pt2 : 1.0f;
    float tmpx = pt0 / denom;
    float tmpy = pt1 / denom;
    float r2 = tmpx * tmpx + tmpy * tmpy;
    float tmpdist = 1.0f + D0 * r2 + D1 * r2 * r2 + D4 * r2 * r2 * r2;
    float ix = tmpx * tmpdist + 2.0f * D2 * tmpx * tmpy + D3 * (r2 + 2.0f * tmpx * tmpx);
    float iy = tmpy * tmpdist + D2 * (r2 + 2.0f * tmpy * tmpy) + 2.0f * D3 * tmpx * tmpy;
    float ux = FX * ix + CX;
    float uy = FY * iy + CY;
    if (close) {
        ox[i] = ux + 0.5f;
        oy[i] = uy + 0.5f;
        oz[i] = pt2 * 100.0f;
        ov[i] = 1;
    } else {
        ox[i] = 0.0f; oy[i] = 0.0f; oz[i] = 0.0f; ov[i] = 0;
    }
}
"#;

const POINT_STEP: usize = 8;

const R00: f32 = -0.9111348390579224;
const R01: f32 =  0.0751304179430008;
const R02: f32 = -0.4052018225193024;
const R10: f32 = -0.3360927104949951;
const R11: f32 = -0.7044632434844971;
const R12: f32 =  0.6251187324523926;
const R20: f32 = -0.2384843230247498;
const R21: f32 =  0.7057529091835022;
const R22: f32 =  0.6671117544174194;
const T0: f32 = 0.1; const T1: f32 = -0.2; const T2: f32 = 0.3;
const D0: f32 = 0.03; const D1: f32 = -0.15; const D2: f32 = 0.001; const D3: f32 = 0.001; const D4: f32 = 0.05;
const FX: f32 = 1200.0; const CX: f32 = 400.0; const FY: f32 = 1200.0; const CY: f32 = 300.0;

fn ref_project(p0: f32, p1: f32, p2: f32) -> (f32, f32, f32, i32) {
    let pt0 = T0 + p0 * R00 + p1 * R01 + p2 * R02;
    let pt1 = T1 + p0 * R10 + p1 * R11 + p2 * R12;
    let pt2 = T2 + p0 * R20 + p1 * R21 + p2 * R22;
    let close = pt2 > 2.5;
    let denom = if close { pt2 } else { 1.0 };
    let tmpx = pt0 / denom;
    let tmpy = pt1 / denom;
    let r2 = tmpx * tmpx + tmpy * tmpy;
    let tmpdist = 1.0 + D0 * r2 + D1 * r2 * r2 + D4 * r2 * r2 * r2;
    let ix = tmpx * tmpdist + 2.0 * D2 * tmpx * tmpy + D3 * (r2 + 2.0 * tmpx * tmpx);
    let iy = tmpy * tmpdist + D2 * (r2 + 2.0 * tmpy * tmpy) + 2.0 * D3 * tmpx * tmpy;
    let ux = FX * ix + CX;
    let uy = FY * iy + CY;
    if close { (ux + 0.5, uy + 0.5, pt2 * 100.0, 1) } else { (0.0, 0.0, 0.0, 0) }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let mut n_batches: i32 = 1;
    let mut i = 1usize;
    while i + 1 < args.len() {
        if args[i] == "-p" { n_batches = args[i+1].parse().unwrap_or(1); }
        i += 1;
    }
    let n_points: usize = 100_000;
    println!("[note] synthetic {} points x {} batches", n_points, n_batches);

    let dev = CudaDevice::new(0)?;
    let ptx = compile_ptx(KERNEL_SRC)?;
    dev.load_ptx(ptx, "daphne", &["project"])?;
    let f = dev.get_func("daphne", "project").unwrap();

    let mut ok_all = true;
    let mut total_us: f64 = 0.0;

    for b in 0..n_batches {
        let mut s: u64 = 20260721u64.wrapping_add((b as u64) + 1);
        let mut cp = vec![0f32; n_points * POINT_STEP];
        for i in 0..n_points {
            let mut u = [0f32; 4];
            for k in 0..4 {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                u[k] = (((s >> 33) & 0x7fffffff) as f32) / 0x7fffffff as f32;
            }
            let base = i * POINT_STEP;
            cp[base]     = (u[0] - 0.5) * 2.0;
            cp[base + 1] = (u[1] - 0.5) * 2.0;
            cp[base + 2] = u[2] * 10.0 + 15.0;
            cp[base + 4] = u[3];
        }

        let d_cp = dev.htod_copy(cp.clone())?;
        let mut d_x = dev.alloc_zeros::<f32>(n_points)?;
        let mut d_y = dev.alloc_zeros::<f32>(n_points)?;
        let mut d_z = dev.alloc_zeros::<f32>(n_points)?;
        let mut d_v = dev.alloc_zeros::<i32>(n_points)?;

        let block: u32 = 256;
        let grid = (n_points as u32 + block - 1) / block;
        let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 };

        dev.synchronize()?;
        let t0 = Instant::now();
        unsafe {
            f.clone().launch(cfg, (&d_cp, &mut d_x, &mut d_y, &mut d_z, &mut d_v, n_points as i32))?;
        }
        dev.synchronize()?;
        total_us += t0.elapsed().as_secs_f64() * 1e6;

        let h_x = dev.dtoh_sync_copy(&d_x)?;
        let h_y = dev.dtoh_sync_copy(&d_y)?;
        let h_z = dev.dtoh_sync_copy(&d_z)?;
        let h_v = dev.dtoh_sync_copy(&d_v)?;

        let mut kept = 0i32;
        let mut mismatch = 0i32;
        let mut max_err: f32 = 0.0;
        for i in 0..n_points {
            let base = i * POINT_STEP;
            let (rx, ry, rz, rv) = ref_project(cp[base], cp[base+1], cp[base+2]);
            if rv == 1 { kept += 1; }
            if h_v[i] != rv { mismatch += 1; continue; }
            if rv == 1 {
                let e = (h_x[i] - rx).abs(); if e > max_err { max_err = e; }
                let e = (h_y[i] - ry).abs(); if e > max_err { max_err = e; }
                let e = (h_z[i] - rz).abs(); if e > max_err { max_err = e; }
            }
        }
        println!("[batch {}] kept={}/{} max_err={:.3e} valid_mismatch={}",
                 b, kept, n_points, max_err, mismatch);
        ok_all = ok_all && mismatch == 0 && max_err <= 1e-3;
    }

    println!("Average kernel execution time: {:.1} (us)", total_us / n_batches as f64);
    println!("{}", if ok_all { "PASS" } else { "FAIL" });
    Ok(())
}
