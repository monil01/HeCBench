use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::env;
use std::time::Instant;

const KERNEL_SRC: &str = r#"
__device__ __forceinline__ float approx_atan2f_P3(float x) {
    return x * (float(-0xf.8eed2p-4) + x * x * float(0x3.1238p-4));
}
__device__ __forceinline__ float approx_atan2f_P5(float x) {
    float z = x*x;
    return x * (float(-0xf.ecfc8p-4) + z * (float(0x4.9e79dp-4) + z * float(-0x1.44f924p-4)));
}
__device__ __forceinline__ float approx_atan2f_P7(float x) {
    float z = x*x;
    return x * (float(-0xf.fcc7ap-4) + z * (float(0x5.23886p-4) + z * (float(-0x2.571968p-4) + z * float(0x9.fb05p-8))));
}
__device__ __forceinline__ float approx_atan2f_P9(float x) {
    float z = x*x;
    return x * (float(-0xf.ff73ep-4) + z * (float(0x5.48ee1p-4) + z * (float(-0x2.e1efe8p-4) + z * (float(0x1.5cce54p-4) + z * float(-0x5.56245p-8)))));
}
__device__ __forceinline__ float approx_atan2f_P11(float x) {
    float z = x*x;
    return x * (float(-0xf.ffe82p-4) + z * (float(0x5.526c8p-4) + z * (float(-0x3.18bea8p-4) + z * (float(0x1.dce3bcp-4) + z * (float(-0xd.7a64ap-8) + z * float(0x3.000eap-8))))));
}
__device__ __forceinline__ float approx_atan2f_P13(float x) {
    float z = x*x;
    return x * (float(-0xf.fffbep-4) + z * (float(0x5.54adp-4) + z * (float(-0x3.2b4df8p-4) + z * (float(0x2.1df79p-4) + z * (float(-0x1.46081p-4) + z * (float(0x8.99028p-8) + z * float(-0x1.be0bc4p-8)))))));
}
__device__ __forceinline__ float approx_atan2f_P15(float x) {
    float z = x*x;
    return x * (float(-0xf.ffff4p-4) + z * (float(0x5.552f9p-4 + z * (float(-0x3.30f728p-4) + z * (float(0x2.39826p-4) + z * (float(-0x1.8a880cp-4) + z * (float(0xe.484d6p-8) + z * (float(-0x5.93d5p-8) + z * float(0x1.0875dcp-8)))))))));
}

__device__ __forceinline__ float safe_atan2f_impl(float y, float x_in, int deg) {
    float x = ((y == 0.f) & (x_in == 0.f)) ? 0.2f : x_in;
    constexpr float pi4f = 3.1415926535897932384626434f / 4.f;
    constexpr float pi34f = 3.1415926535897932384626434f * 3.f / 4.f;
    float ax = fabsf(x);
    float ay = fabsf(y);
    float r = (ax - ay) / (ax + ay);
    if (x < 0) r = -r;
    float angle = (x >= 0) ? pi4f : pi34f;
    float p;
    switch (deg) {
      case 3:  p = approx_atan2f_P3(r); break;
      case 5:  p = approx_atan2f_P5(r); break;
      case 7:  p = approx_atan2f_P7(r); break;
      case 9:  p = approx_atan2f_P9(r); break;
      case 11: p = approx_atan2f_P11(r); break;
      case 13: p = approx_atan2f_P13(r); break;
      default: p = approx_atan2f_P15(r); break;
    }
    angle += p;
    return (y < 0) ? -angle : angle;
}

extern "C" __global__ void compute_f(const int n, const float* x, const float* y, float* r) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float vy = y[i];
    const float vx = x[i];
    r[i] = safe_atan2f_impl(vy, vx, 3) +
           safe_atan2f_impl(vy, vx, 5) +
           safe_atan2f_impl(vy, vx, 7) +
           safe_atan2f_impl(vy, vx, 9) +
           safe_atan2f_impl(vy, vx, 11) +
           safe_atan2f_impl(vy, vx, 13) +
           safe_atan2f_impl(vy, vx, 15);
}
"#;

use hexf::hexf32;

fn p3f(x: f32) -> f32 {
    x * (hexf32!("-0xf.8eed2p-4") + x*x*hexf32!("0x3.1238p-4"))
}
fn p5f(x: f32) -> f32 {
    let z = x*x;
    x * (hexf32!("-0xf.ecfc8p-4") + z*(hexf32!("0x4.9e79dp-4") + z*hexf32!("-0x1.44f924p-4")))
}
fn p7f(x: f32) -> f32 {
    let z = x*x;
    x * (hexf32!("-0xf.fcc7ap-4") + z*(hexf32!("0x5.23886p-4") + z*(hexf32!("-0x2.571968p-4") + z*hexf32!("0x9.fb05p-8"))))
}
fn p9f(x: f32) -> f32 {
    let z = x*x;
    x * (hexf32!("-0xf.ff73ep-4") + z*(hexf32!("0x5.48ee1p-4") + z*(hexf32!("-0x2.e1efe8p-4") + z*(hexf32!("0x1.5cce54p-4") + z*hexf32!("-0x5.56245p-8")))))
}
fn p11f(x: f32) -> f32 {
    let z = x*x;
    x * (hexf32!("-0xf.ffe82p-4") + z*(hexf32!("0x5.526c8p-4") + z*(hexf32!("-0x3.18bea8p-4") + z*(hexf32!("0x1.dce3bcp-4") + z*(hexf32!("-0xd.7a64ap-8") + z*hexf32!("0x3.000eap-8"))))))
}
fn p13f(x: f32) -> f32 {
    let z = x*x;
    x * (hexf32!("-0xf.fffbep-4") + z*(hexf32!("0x5.54adp-4") + z*(hexf32!("-0x3.2b4df8p-4") + z*(hexf32!("0x2.1df79p-4") + z*(hexf32!("-0x1.46081p-4") + z*(hexf32!("0x8.99028p-8") + z*hexf32!("-0x1.be0bc4p-8")))))))
}
// P15 - matches the exact (buggy) nesting in CUDA source
fn p15f(x: f32) -> f32 {
    let z = x*x;
    // In the C source: x * (float(-0xf.ffff4p-4) + z * (float(0x5.552f9p-4 + z * (float(-0x3.30f728p-4) + z * ...))))
    // The outer parentheses group `float(0x5.552f9p-4 + z * (...))` which means the entire expression
    // in the middle is: 0x5.552f9p-4 + z * (rest)
    let c0 = hexf32!("-0xf.ffff4p-4");
    let c1 = hexf32!("0x5.552f9p-4");
    let c2 = hexf32!("-0x3.30f728p-4");
    let c3 = hexf32!("0x2.39826p-4");
    let c4 = hexf32!("-0x1.8a880cp-4");
    let c5 = hexf32!("0xe.484d6p-8");
    let c6 = hexf32!("-0x5.93d5p-8");
    let c7 = hexf32!("0x1.0875dcp-8");
    let deep = c2 + z*(c3 + z*(c4 + z*(c5 + z*(c6 + z*c7))));
    let middle = c1 + z*deep;
    x * (c0 + z*middle)
}

fn safe_atan2f_rs(y: f32, x_in: f32, deg: i32) -> f32 {
    let x = if y == 0.0 && x_in == 0.0 { 0.2f32 } else { x_in };
    let pi4f: f32 = std::f32::consts::PI / 4.0;
    let pi34f: f32 = std::f32::consts::PI * 3.0 / 4.0;
    let ax = x.abs();
    let ay = y.abs();
    let mut r = (ax - ay) / (ax + ay);
    if x < 0.0 { r = -r; }
    let mut angle = if x >= 0.0 { pi4f } else { pi34f };
    let p = match deg { 3=>p3f(r), 5=>p5f(r), 7=>p7f(r), 9=>p9f(r), 11=>p11f(r), 13=>p13f(r), _=>p15f(r) };
    angle += p;
    if y < 0.0 { -angle } else { angle }
}

fn reference_f(x: &[f32], y: &[f32], r: &mut [f32]) {
    for i in 0..x.len() {
        let vy = y[i]; let vx = x[i];
        r[i] = safe_atan2f_rs(vy, vx, 3)
             + safe_atan2f_rs(vy, vx, 5)
             + safe_atan2f_rs(vy, vx, 7)
             + safe_atan2f_rs(vy, vx, 9)
             + safe_atan2f_rs(vy, vx, 11)
             + safe_atan2f_rs(vy, vx, 13)
             + safe_atan2f_rs(vy, vx, 15);
    }
}

// simple LCG matching C rand() with seed 123 - use libc? We can't. Use fixed seed rng via std
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 { eprintln!("Usage: {} <n> <repeat>", args[0]); std::process::exit(1); }
    let n: usize = args[1].parse()?;
    let repeat: i32 = args[2].parse()?;

    let mut x = vec![0f32; n];
    let mut y = vec![0f32; n];
    // Simple xorshift RNG for reproducibility
    let mut state: u64 = 123456789;
    let mut next_f = || -> f32 {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        ((state as u32) & 0x00FFFFFF) as f32 / (1<<24) as f32
    };
    for i in 0..n { x[i] = next_f() + 1.57; y[i] = next_f() + 1.57; }

    let dev = CudaDevice::new(0)?;
    let ptx = compile_ptx(KERNEL_SRC)?;
    dev.load_ptx(ptx, "m", &["compute_f"])?;
    let f = dev.get_func("m", "compute_f").unwrap();

    let d_x = dev.htod_copy(x.clone())?;
    let d_y = dev.htod_copy(y.clone())?;
    let mut d_r = dev.alloc_zeros::<f32>(n)?;

    let cfg = LaunchConfig::for_num_elems(n as u32);

    dev.synchronize()?;
    let start = Instant::now();
    for _ in 0..repeat {
        unsafe { f.clone().launch(cfg, (n as i32, &d_y, &d_x, &mut d_r))?; }
    }
    dev.synchronize()?;
    let elapsed = start.elapsed();
    println!("Average execution time (f32): {} (us)", elapsed.as_secs_f64() * 1e6 / repeat as f64);

    let h_gpu: Vec<f32> = dev.dtoh_sync_copy(&d_r)?;
    let mut h_cpu = vec![0f32; n];
    // Match CUDA main: reference_f (n, y, x, rf) — swap arg order
    reference_f(&y, &x, &mut h_cpu);

    let mut err = 0f64;
    let mut max_diff = 0f32;
    for i in 0..n {
        let d = (h_gpu[i] - h_cpu[i]).abs();
        if d > max_diff { max_diff = d; }
        if d > 1e-3 { err += (d as f64)*(d as f64); }
    }
    let rmse = (err / n as f64).sqrt();
    println!("RMSE: {:e}, max_abs_diff: {:e}", rmse, max_diff);
    let ok = max_diff < 1e-2;
    println!("{}", if ok { "PASS" } else { "FAIL" });

    Ok(())
}
