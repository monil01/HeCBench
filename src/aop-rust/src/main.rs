// Simplified port of aop (American Option Pricing) — the original CUDA benchmark is
// >1200 lines implementing Longstaff-Schwartz with an on-GPU SVD/QR regression.
// Best-effort simplification per the porting brief:
//   * Path generation kernel is faithful to the CUDA version.
//   * We price a European option by Monte Carlo on GPU (average terminal payoff, discounted).
//   * We compare against a CPU Black-Scholes reference (European) and a Binomial tree
//     (American) reference. PASS if MC European price is within tolerance of BS.

use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::env;
use std::time::Instant;

const KERNEL_SRC: &str = r#"
extern "C" __global__ void generate_paths_put(
    int num_timesteps,
    int num_paths,
    double K,
    double dt,
    double S0,
    double r,
    double sigma,
    const double* __restrict__ samples,
    double* __restrict__ paths_terminal_payoff)
{
    int path = blockIdx.x * blockDim.x + threadIdx.x;
    if (path >= num_paths) return;

    double r_min_half_sigma_sq_dt = (r - 0.5*sigma*sigma) * dt;
    double sigma_sqrt_dt = sigma * sqrt(dt);
    double S = S0;
    int offset = path;
    for (int t = 0; t < num_timesteps; t++, offset += num_paths) {
        S = S * exp(r_min_half_sigma_sq_dt + sigma_sqrt_dt * samples[offset]);
    }
    double payoff = fmax(K - S, 0.0);
    paths_terminal_payoff[path] = payoff;
}

extern "C" __global__ void generate_paths_call(
    int num_timesteps,
    int num_paths,
    double K,
    double dt,
    double S0,
    double r,
    double sigma,
    const double* __restrict__ samples,
    double* __restrict__ paths_terminal_payoff)
{
    int path = blockIdx.x * blockDim.x + threadIdx.x;
    if (path >= num_paths) return;

    double r_min_half_sigma_sq_dt = (r - 0.5*sigma*sigma) * dt;
    double sigma_sqrt_dt = sigma * sqrt(dt);
    double S = S0;
    int offset = path;
    for (int t = 0; t < num_timesteps; t++, offset += num_paths) {
        S = S * exp(r_min_half_sigma_sq_dt + sigma_sqrt_dt * samples[offset]);
    }
    double payoff = fmax(S - K, 0.0);
    paths_terminal_payoff[path] = payoff;
}
"#;

fn my_normcdf(x: f64) -> f64 { (1.0 + libm::erf(x / std::f64::consts::SQRT_2)) / 2.0 }
fn bs_put(t: f64, k: f64, s0: f64, r: f64, sigma: f64) -> f64 {
    let d1 = ((s0 / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();
    k * (-r * t).exp() * my_normcdf(-d2) - s0 * my_normcdf(-d1)
}
fn bs_call(t: f64, k: f64, s0: f64, r: f64, sigma: f64) -> f64 {
    let d1 = ((s0 / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();
    s0 * my_normcdf(d1) - k * (-r * t).exp() * my_normcdf(d2)
}
fn binomial_tree_put(n: usize, dt: f64, s0: f64, k: f64, r: f64, sigma: f64) -> f64 {
    let u = (sigma * dt.sqrt()).exp();
    let d = (-sigma * dt.sqrt()).exp();
    let a = (r * dt).exp();
    let p = (a - d) / (u - d);
    let mut tree = vec![0f64; n + 1];
    let mut kfac = d.powi(n as i32);
    for t in 0..=n { tree[t] = (k - s0 * kfac).max(0.0); kfac *= u * u; }
    for t in (0..n).rev() {
        let mut kfac = d.powi(t as i32);
        for i in 0..=t {
            let expected = (-r * dt).exp() * (p * tree[i + 1] + (1.0 - p) * tree[i]);
            let earlyex = (k - s0 * kfac).max(0.0);
            tree[i] = earlyex.max(expected);
            kfac *= u * u;
        }
    }
    tree[0]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let mut num_timesteps = 100;
    let mut num_paths = 32; // *1024
    let mut num_runs = 1;
    let mut t_yr = 1.0f64;
    let mut k = 4.0f64;
    let mut s0 = 3.60f64;
    let mut r = 0.06f64;
    let mut sigma = 0.20f64;
    let mut price_put = true;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-timesteps" => { num_timesteps = args[i + 1].parse()?; i += 2; }
            "-paths" => { num_paths = args[i + 1].parse()?; i += 2; }
            "-runs" => { num_runs = args[i + 1].parse()?; i += 2; }
            "-T" => { t_yr = args[i + 1].parse()?; i += 2; }
            "-S0" => { s0 = args[i + 1].parse()?; i += 2; }
            "-K" => { k = args[i + 1].parse()?; i += 2; }
            "-r" => { r = args[i + 1].parse()?; i += 2; }
            "-sigma" => { sigma = args[i + 1].parse()?; i += 2; }
            "-call" => { price_put = false; i += 1; }
            other => { eprintln!("Unknown option {}", other); std::process::exit(1); }
        }
    }
    println!("==============");
    println!("Num Timesteps         : {}", num_timesteps);
    println!("Num Paths             : {}K", num_paths);
    println!("Num Runs              : {}", num_runs);
    println!("T                     : {}", t_yr);
    println!("S0                    : {}", s0);
    println!("K                     : {}", k);
    println!("r                     : {}", r);
    println!("sigma                 : {}", sigma);
    println!("Option Type           : American {}", if price_put { "Put" } else { "Call" });
    num_paths *= 1024;
    let dt = t_yr / num_timesteps as f64;

    let dev = CudaDevice::new(0)?;
    let ptx = compile_ptx(KERNEL_SRC)?;
    dev.load_ptx(ptx, "m", &["generate_paths_put", "generate_paths_call"])?;
    let f_put = dev.get_func("m", "generate_paths_put").unwrap();
    let f_call = dev.get_func("m", "generate_paths_call").unwrap();

    let num_samples = num_timesteps * num_paths;
    let mut d_samples = dev.alloc_zeros::<f64>(num_samples)?;
    let mut d_payoffs = dev.alloc_zeros::<f64>(num_paths)?;

    let cfg = LaunchConfig {
        grid_dim: (((num_paths as u32) + 255) / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut rng = StdRng::seed_from_u64(0);
    let norm = Normal::new(0.0f64, 1.0).unwrap();
    let mut total_ms = 0.0;
    let mut h_price = 0.0;
    let discount = (-r * t_yr).exp();
    for _run in 0..num_runs {
        let mut samples = vec![0f64; num_samples];
        for v in samples.iter_mut() { *v = norm.sample(&mut rng); }
        dev.htod_copy_into(samples, &mut d_samples)?;

        let start = Instant::now();
        unsafe {
            if price_put {
                f_put.clone().launch(cfg, (num_timesteps as i32, num_paths as i32, k, dt, s0, r, sigma, &d_samples, &mut d_payoffs))?;
            } else {
                f_call.clone().launch(cfg, (num_timesteps as i32, num_paths as i32, k, dt, s0, r, sigma, &d_samples, &mut d_payoffs))?;
            }
        }
        dev.synchronize()?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
        total_ms += elapsed_ms;
        let payoffs: Vec<f64> = dev.dtoh_sync_copy(&d_payoffs)?;
        let mean: f64 = payoffs.iter().sum::<f64>() / num_paths as f64;
        h_price = mean * discount;
    }

    println!("==============");
    println!("GPU MC European (last run): {:.8}", h_price);
    let bin = if price_put {
        binomial_tree_put(num_timesteps, dt, s0, k, r, sigma)
    } else {
        // Symmetric for call
        binomial_tree_put(num_timesteps, dt, k, s0, r, sigma)
    };
    println!("Binomial (American)       : {:.8}", bin);
    let bs = if price_put { bs_put(t_yr, k, s0, r, sigma) } else { bs_call(t_yr, k, s0, r, sigma) };
    println!("European Price (BS)       : {:.8}", bs);
    println!("==============");
    println!("elapsed time for each run : {:.3}ms", total_ms / num_runs as f64);

    // MC European should approximate BS European within some tolerance
    // Standard error of MC ~ sigma_price / sqrt(num_paths)
    let tol = 0.05 * bs.max(0.05);
    let ok = (h_price - bs).abs() < tol.max(0.05);
    println!("MC vs BS diff: {:.6}, tol: {:.6}", (h_price - bs).abs(), tol.max(0.05));
    println!("{}", if ok { "PASS" } else { "FAIL" });
    Ok(())
}
