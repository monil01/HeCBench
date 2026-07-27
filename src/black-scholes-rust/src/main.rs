// Black-Scholes analytic pricing. Simplified from black-scholes-cuda:
// instead of the QuantLib-style optionInputStruct with dividends, we price
// vanilla European calls and puts using the closed-form B-S formula on a
// random sample. The CPU reference recomputes the same formula with libm.
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::env;
use std::time::Instant;

const KERNEL_SRC: &str = r#"
extern "C" __global__ void bs_price(
    const float* __restrict__ S,
    const float* __restrict__ K,
    const float* __restrict__ T,
    const float* __restrict__ r,
    const float* __restrict__ sigma,
    const int* __restrict__ is_put,
    float* __restrict__ price,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = S[i], k = K[i], t = T[i], rr = r[i], sig = sigma[i];
    float sqrtT = sqrtf(t);
    float d1 = (logf(s / k) + (rr + 0.5f * sig * sig) * t) / (sig * sqrtT);
    float d2 = d1 - sig * sqrtT;
    // N(x) via erf
    float Nd1 = 0.5f * (1.0f + erff(d1 * 0.70710678118654752440f));
    float Nd2 = 0.5f * (1.0f + erff(d2 * 0.70710678118654752440f));
    float call = s * Nd1 - k * expf(-rr * t) * Nd2;
    if (is_put[i]) {
        // put-call parity: put = call - S + K exp(-rT)
        price[i] = call - s + k * expf(-rr * t);
    } else {
        price[i] = call;
    }
}
"#;

fn cpu_price(s: f32, k: f32, t: f32, r: f32, sig: f32, put: bool) -> f32 {
    let sqrtt = t.sqrt();
    let d1 = ((s / k).ln() + (r + 0.5 * sig * sig) * t) / (sig * sqrtt);
    let d2 = d1 - sig * sqrtt;
    let nd1 = 0.5 * (1.0 + libm::erff(d1 * std::f32::consts::FRAC_1_SQRT_2));
    let nd2 = 0.5 * (1.0 + libm::erff(d2 * std::f32::consts::FRAC_1_SQRT_2));
    let call = s * nd1 - k * (-r * t).exp() * nd2;
    if put {
        call - s + k * (-r * t).exp()
    } else {
        call
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let repeat: i32 = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(100);
    let n: usize = 1 << 20; // 1M options

    let mut rng = StdRng::seed_from_u64(0xB5C4);
    let s: Vec<f32> = (0..n).map(|_| rng.gen_range(50.0..150.0)).collect();
    let k: Vec<f32> = (0..n).map(|_| rng.gen_range(50.0..150.0)).collect();
    let t: Vec<f32> = (0..n).map(|_| rng.gen_range(0.05..1.0)).collect();
    let r: Vec<f32> = (0..n).map(|_| rng.gen_range(0.01..0.1)).collect();
    let sig: Vec<f32> = (0..n).map(|_| rng.gen_range(0.1..0.5)).collect();
    let put: Vec<i32> = (0..n).map(|_| rng.gen_range(0..=1)).collect();

    let dev = CudaDevice::new(0)?;
    let ptx = compile_ptx(KERNEL_SRC)?;
    dev.load_ptx(ptx, "bs", &["bs_price"])?;
    let f = dev.get_func("bs", "bs_price").unwrap();

    let d_s = dev.htod_copy(s.clone())?;
    let d_k = dev.htod_copy(k.clone())?;
    let d_t = dev.htod_copy(t.clone())?;
    let d_r = dev.htod_copy(r.clone())?;
    let d_sig = dev.htod_copy(sig.clone())?;
    let d_put = dev.htod_copy(put.clone())?;
    let mut d_price = dev.alloc_zeros::<f32>(n)?;

    let block = 256u32;
    let grid = ((n as u32) + block - 1) / block;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };

    dev.synchronize()?;
    let start = Instant::now();
    for _ in 0..repeat {
        unsafe {
            f.clone().launch(cfg, (&d_s, &d_k, &d_t, &d_r, &d_sig, &d_put, &mut d_price, n as i32))?;
        }
    }
    dev.synchronize()?;
    let elapsed = start.elapsed();
    println!("Average kernel time: {:.3} (ms)",
             elapsed.as_secs_f64() * 1000.0 / repeat as f64);

    let gpu: Vec<f32> = dev.dtoh_sync_copy(&d_price)?;
    let mut ok = true;
    let mut worst: f32 = 0.0;
    for i in 0..n {
        let expected = cpu_price(s[i], k[i], t[i], r[i], sig[i], put[i] != 0);
        let diff = (gpu[i] - expected).abs();
        let tol = 1e-3_f32 * expected.abs().max(1.0);
        worst = worst.max(diff / expected.abs().max(1.0));
        if diff > tol {
            println!("Mismatch @ {}: gpu={} cpu={}", i, gpu[i], expected);
            ok = false;
            break;
        }
    }
    let sum: f64 = gpu.iter().map(|&x| x as f64).sum();
    println!("Sum GPU prices: {}", sum);
    println!("Worst rel err: {:.3e}", worst);
    println!("{}", if ok { "PASS" } else { "FAIL" });
    Ok(())
}
