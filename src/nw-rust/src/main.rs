// Needleman-Wunsch, simple per-anti-diagonal wavefront (one thread per cell).
// Reference computed in Rust from the same input; PASS if match exactly.
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::env;
use std::time::Instant;

const KERNEL_SRC: &str = r#"
extern "C" __global__ void nw_diag(int* itemsets, const int* reference,
                                   int max_cols, int diag, int penalty,
                                   int i_min, int i_max)
{
    // Cell (i, j) with i + j == diag. Thread walks over valid i.
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int i = i_min + t;
    if (i > i_max) return;
    int j = diag - i;
    int idx = i * max_cols + j;
    int nw_v = itemsets[(i-1) * max_cols + (j-1)] + reference[idx];
    int w    = itemsets[i * max_cols + (j-1)] - penalty;
    int n    = itemsets[(i-1) * max_cols + j] - penalty;
    int r = nw_v;
    if (w > r) r = w;
    if (n > r) r = n;
    itemsets[idx] = r;
}
"#;

const BLOSUM62: [[i32; 24]; 24] = [
    [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0,-2,-1, 0,-4],
    [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3,-1, 0,-1,-4],
    [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3, 3, 0,-1,-4],
    [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3, 4, 1,-1,-4],
    [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1,-3,-3,-2,-4],
    [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2, 0, 3,-1,-4],
    [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2, 1, 4,-1,-4],
    [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3,-1,-2,-1,-4],
    [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3, 0, 0,-1,-4],
    [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3,-3,-3,-1,-4],
    [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1,-4,-3,-1,-4],
    [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2, 0, 1,-1,-4],
    [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1,-3,-1,-1,-4],
    [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1,-3,-3,-1,-4],
    [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2,-2,-1,-2,-4],
    [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2, 0, 0, 0,-4],
    [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0,-1,-1, 0,-4],
    [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3,-4,-3,-2,-4],
    [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1,-3,-2,-1,-4],
    [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4,-3,-2,-1,-4],
    [-2,-1, 3, 4,-3, 0, 1,-1, 0,-3,-4, 0,-3,-3,-2, 0,-1,-4,-3,-3, 4, 1,-1,-4],
    [-1, 0, 0, 1,-3, 3, 4,-2, 0,-3,-3, 1,-1,-3,-1, 0,-1,-3,-2,-2, 1, 4,-1,-4],
    [ 0,-1,-1,-1,-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-2, 0, 0,-2,-1,-1,-1,-1,-1,-4],
    [-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4, 1],
];

fn nw_host(itemsets: &mut [i32], reference: &[i32], max_cols: usize, penalty: i32) {
    for i in 1..max_cols {
        for j in 1..max_cols {
            let idx = i * max_cols + j;
            let nw_v = itemsets[(i-1) * max_cols + (j-1)] + reference[idx];
            let w    = itemsets[i * max_cols + (j-1)] - penalty;
            let n    = itemsets[(i-1) * max_cols + j] - penalty;
            itemsets[idx] = nw_v.max(w).max(n);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let dim: usize = if args.len() > 1 { args[1].parse()? } else { 256 };
    let penalty: i32 = if args.len() > 2 { args[2].parse()? } else { 10 };
    let repeat: i32 = if args.len() > 3 { args[3].parse()? } else { 10 };

    let max_cols = dim + 1;
    let max_rows = dim + 1;

    let mut rng = StdRng::seed_from_u64(7);
    let mut itemsets = vec![0i32; max_rows * max_cols];
    let mut reference = vec![0i32; max_rows * max_cols];
    for i in 1..max_rows { itemsets[i * max_cols] = (rng.r#gen::<u32>() % 10 + 1) as i32; }
    for j in 1..max_cols { itemsets[j]            = (rng.r#gen::<u32>() % 10 + 1) as i32; }
    for i in 1..max_cols { for j in 1..max_rows {
        let a = itemsets[i * max_cols] as usize;
        let b = itemsets[j] as usize;
        reference[i * max_cols + j] = BLOSUM62[a][b];
    }}
    for i in 1..max_rows { itemsets[i * max_cols] = -(i as i32) * penalty; }
    for j in 1..max_cols { itemsets[j]            = -(j as i32) * penalty; }
    let init = itemsets.clone();

    let dev = CudaDevice::new(0)?;
    let ptx = compile_ptx(KERNEL_SRC)?;
    dev.load_ptx(ptx, "nw", &["nw_diag"])?;
    let f = dev.get_func("nw", "nw_diag").unwrap();

    let mut d_items = dev.htod_copy(itemsets.clone())?;
    let d_ref = dev.htod_copy(reference.clone())?;

    dev.synchronize()?;
    let start = Instant::now();
    for _ in 0..repeat {
        dev.htod_copy_into(init.clone(), &mut d_items)?;
        let n = dim;
        for diag in 2..=(2 * n) {
            let i_min = if diag > n { diag - n } else { 1 };
            let i_max = if diag > n { n } else { diag - 1 };
            let count = (i_max - i_min + 1) as u32;
            let block = 128u32;
            let grid = (count + block - 1) / block;
            let cfg = LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                f.clone().launch(cfg, (&mut d_items, &d_ref,
                                       max_cols as i32, diag as i32,
                                       penalty, i_min as i32, i_max as i32))?;
            }
        }
    }
    dev.synchronize()?;
    let elapsed = start.elapsed();
    println!("Total kernel execution time: {:.6} (s)", elapsed.as_secs_f64() / repeat as f64);

    let out: Vec<i32> = dev.dtoh_sync_copy(&d_items)?;
    let mut ref_out = init.clone();
    nw_host(&mut ref_out, &reference, max_cols, penalty);

    let mut ok = true;
    for k in 0..out.len() {
        if out[k] != ref_out[k] { ok = false; break; }
    }
    println!("{}", if ok { "PASS" } else { "FAIL" });
    Ok(())
}
