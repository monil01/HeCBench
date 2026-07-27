use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::env;
use std::time::Instant;

const KERNEL_SRC: &str = r#"
#define GPU_NUM_THREADS 256

// simple shared-memory block sum reduction; only thread 0 has the final sum
__device__ int blockSumInt(int val) {
    __shared__ int sdata[GPU_NUM_THREADS];
    int tid = threadIdx.x;
    sdata[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    return sdata[0];
}

extern "C" __global__ void accuracy_kernel(
    const int N,
    const int D,
    const int top_k,
    const float* __restrict__ Xdata,
    const int*   __restrict__ labelData,
    int* accuracy)
{
    int count = 0;
    for (int row = blockIdx.x; row < N; row += gridDim.x) {
        const int label = labelData[row];
        const float label_pred = Xdata[row * D + label];
        int ngt = 0;
        for (int col = threadIdx.x; col < D; col += blockDim.x) {
            const float pred = Xdata[row * D + col];
            if (pred > label_pred || (pred == label_pred && col <= label)) {
                ++ngt;
            }
        }
        int sum_ngt = blockSumInt(ngt);
        if (threadIdx.x == 0 && sum_ngt <= top_k) {
            ++count;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(accuracy, count);
    }
}
"#;

fn reference(nrows: usize, ndims: usize, top_k: i32, xdata: &[f32], label: &[i32]) -> i32 {
    let mut count = 0i32;
    for row in 0..nrows {
        let lab = label[row] as usize;
        let label_pred = xdata[row * ndims + lab];
        let mut ngt = 0i32;
        for col in 0..ndims {
            let pred = xdata[row * ndims + col];
            if pred > label_pred || (pred == label_pred && (col as i32) <= (lab as i32)) {
                ngt += 1;
            }
        }
        if ngt <= top_k {
            count += 1;
        }
    }
    count
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 5 {
        eprintln!("Usage: {} <rows> <cols> <top_k> <repeat>", args[0]);
        std::process::exit(1);
    }
    let nrows: usize = args[1].parse()?;
    let ndims: usize = args[2].parse()?;
    let top_k: i32 = args[3].parse()?;
    let repeat: i32 = args[4].parse()?;

    let data_size = nrows * ndims;

    let mut rng = StdRng::seed_from_u64(123);
    let label: Vec<i32> = (0..nrows).map(|_| rng.gen_range(0..ndims as i32)).collect();
    let data: Vec<f32> = (0..data_size).map(|_| rng.r#gen::<f32>()).collect();

    let count_ref = reference(nrows, ndims, top_k, &data, &label);
    println!("Reference count: {}", count_ref);

    let dev = CudaDevice::new(0)?;
    let ptx = compile_ptx(KERNEL_SRC)?;
    dev.load_ptx(ptx, "acc", &["accuracy_kernel"])?;
    let f = dev.get_func("acc", "accuracy_kernel").unwrap();

    let d_label = dev.htod_copy(label.clone())?;
    let d_data = dev.htod_copy(data.clone())?;

    let block_dim = 256u32;

    let mut ngrid = nrows / 4;
    while ngrid <= nrows {
        println!("Grid size is {}", ngrid);
        let cfg = LaunchConfig {
            grid_dim: (ngrid as u32, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };

        dev.synchronize()?;
        let start = Instant::now();
        let mut d_count = dev.alloc_zeros::<i32>(1)?;
        for _ in 0..repeat {
            dev.memset_zeros(&mut d_count)?;
            unsafe {
                f.clone().launch(
                    cfg,
                    (
                        nrows as i32,
                        ndims as i32,
                        top_k,
                        &d_data,
                        &d_label,
                        &mut d_count,
                    ),
                )?;
            }
        }
        dev.synchronize()?;
        let elapsed = start.elapsed();
        println!(
            "Average execution time of accuracy kernel: {} (us)",
            elapsed.as_secs_f64() * 1e6 / repeat as f64
        );
        let out: Vec<i32> = dev.dtoh_sync_copy(&d_count)?;
        let ok = out[0] == count_ref;
        println!("GPU count: {} vs ref: {}", out[0], count_ref);
        println!("{}", if ok { "PASS" } else { "FAIL" });

        ngrid += nrows / 4;
    }

    Ok(())
}
