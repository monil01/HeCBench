// Rust port of the `heat2d` HeCBench benchmark.
//
// A single Laplace-relaxation iteration on a torus: each site gets a
// weighted sum of its four neighbours + itself. Runs `niter` iterations
// on the GPU (kernel is nearly the CUDA source verbatim, compiled via
// NVRTC) and compares to a Rust CPU reference to within 1e-2 abs err
// (matches the CUDA benchmark's declared tolerance — error grows with
// iteration count).
//
// Usage: heat2d-rust <Lx> <Ly> <niter>
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::env;
use std::time::Instant;

const NTX: usize = 16;
const NTY: usize = 16;

const KERNEL_SRC: &str = r#"
extern "C" __global__ void
dev_lapl_iter(float *out, const float *in, const float delta, const float norm,
              const int Lx, const int Ly)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int x = i % Lx;
    int y = i / Lx;
    int v00 = y*Lx + x;
    int v0p = y*Lx + (x + 1)%Lx;
    int v0m = y*Lx + (Lx + x - 1)%Lx;
    int vp0 = ((y+1)%Ly)*Lx + x;
    int vm0 = ((Ly+y-1)%Ly)*Lx + x;
    out[v00] = norm*in[v00] + delta*(in[v0p] + in[v0m] + in[vp0] + in[vm0]);
}
"#;

fn reference(out: &mut [f32], input: &[f32], delta: f32, norm: f32, lx: usize, ly: usize) {
    for y in 0..ly {
        for x in 0..lx {
            let v00 = y*lx + x;
            let v0p = y*lx + (x + 1) % lx;
            let v0m = y*lx + (lx + x - 1) % lx;
            let vp0 = ((y + 1) % ly)*lx + x;
            let vm0 = ((ly + y - 1) % ly)*lx + x;
            out[v00] = norm*input[v00] + delta*(input[v0p] + input[v0m] + input[vp0] + input[vm0]);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: {} <Lx> <Ly> <niter>", args[0]);
        std::process::exit(1);
    }
    let lx: usize = args[1].parse()?;
    let ly: usize = args[2].parse()?;
    let niter: usize = args[3].parse()?;
    if lx % NTX != 0 || ly % NTY != 0 {
        eprintln!("Lx must be a multiple of {}, Ly a multiple of {}", NTX, NTY);
        std::process::exit(1);
    }

    let sigma = 0.01f32;
    let xdelta = sigma / (1.0 + 4.0*sigma);
    let xnorm = 1.0 / (1.0 + 4.0*sigma);

    println!(" Ly,Lx = {},{}", ly, lx);
    println!(" niter = {}", niter);

    // Match CUDA init: sparse row/column strokes at random positions, seed 123.
    let mut rng = StdRng::seed_from_u64(123);
    let mut buffer = vec![0.0f32; lx*ly];
    for _i in (0..lx).step_by(16) {
        let x: usize = rng.gen_range(0..lx);
        for j in 0..ly {
            buffer[x + j*lx] = 1.0;
        }
    }
    for _i in (0..ly).step_by(16) {
        let y: usize = rng.gen_range(0..ly);
        for j in 0..lx {
            buffer[j + y*lx] = 1.0;
        }
    }

    // CPU reference: niter iterations of the same stencil
    let mut h_in = buffer.clone();
    let mut h_out = vec![0.0f32; lx*ly];
    for _ in 0..niter {
        reference(&mut h_out, &h_in, xdelta, xnorm, lx, ly);
        std::mem::swap(&mut h_in, &mut h_out);
    }

    // GPU path
    let dev = CudaDevice::new(0)?;
    let ptx = compile_ptx(KERNEL_SRC)?;
    dev.load_ptx(ptx, "heat2d", &["dev_lapl_iter"])?;
    let f = dev.get_func("heat2d", "dev_lapl_iter").unwrap();

    let mut d_in = dev.htod_copy(buffer.clone())?;
    let mut d_out = dev.alloc_zeros::<f32>(lx*ly)?;

    let block = (NTX * NTY) as u32;
    let grid  = ((lx / NTX) * (ly / NTY)) as u32;
    let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 };

    dev.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..niter {
        unsafe { f.clone().launch(cfg, (&mut d_out, &d_in, xdelta, xnorm, lx as i32, ly as i32))?; }
        std::mem::swap(&mut d_in, &mut d_out);
    }
    dev.synchronize()?;
    let elapsed = t0.elapsed();
    let per_iter_us = elapsed.as_nanos() as f64 * 1e-3 / niter as f64;
    let bw_gbs = (lx*ly) as f64 * 4.0 * 2.0 / (per_iter_us * 1000.0);
    let gflops = (lx*ly) as f64 * 6.0 / (per_iter_us * 1000.0);
    println!(
        "Device: iters = {:8}, (Lx,Ly) = {:6}, {:6}, t = {:8.1} usec/iter, BW = {:6.3} GB/s, P = {:6.3} Gflop/s",
        niter, lx, ly, per_iter_us, bw_gbs, gflops
    );

    let d_res: Vec<f32> = dev.dtoh_sync_copy(&d_in)?;

    let mut ok = true;
    for i in 0..lx*ly {
        if (h_in[i] - d_res[i]).abs() > 1e-2 {
            println!("Mismatch at {} cpu={} gpu={}", i, h_in[i], d_res[i]);
            ok = false;
            break;
        }
    }
    println!("{}", if ok { "PASS" } else { "FAIL" });
    Ok(())
}
