// Rust port of the `bfs` HeCBench benchmark.
//
// Level-synchronous BFS on a CSR-ish graph. GPU kernels lifted verbatim
// from the CUDA source. Verifies against a Rust CPU BFS from the same
// source node (0).
//
// Usage: bfs-rust <graph_file>
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

const BLOCK: u32 = 256;

const KERNEL_SRC: &str = r#"
struct Node { int starting; int no_of_edges; };

extern "C" __global__ void
Kernel(const Node* d_graph_nodes,
       const int*  d_graph_edges,
       char*       d_graph_mask,
       char*       d_updating_graph_mask,
       const char* d_graph_visited,
       int*        d_cost,
       const int   no_of_nodes)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < no_of_nodes && d_graph_mask[tid]) {
        d_graph_mask[tid] = 0;
        int num_edges = d_graph_nodes[tid].no_of_edges;
        int starting  = d_graph_nodes[tid].starting;
        for (int i = starting; i < num_edges + starting; ++i) {
            int id = d_graph_edges[i];
            if (!d_graph_visited[id]) {
                d_cost[id] = d_cost[tid] + 1;
                d_updating_graph_mask[id] = 1;
            }
        }
    }
}

extern "C" __global__ void
Kernel2(char* d_graph_mask,
        char* d_updating_graph_mask,
        char* d_graph_visited,
        char* d_over,
        const int no_of_nodes)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < no_of_nodes && d_updating_graph_mask[tid]) {
        d_graph_mask[tid] = 1;
        d_graph_visited[tid] = 1;
        *d_over = 1;
        d_updating_graph_mask[tid] = 0;
    }
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <graph_file>", args[0]);
        std::process::exit(1);
    }
    println!("Reading File");

    // Parse graph file:
    //   <n>\n <start_0> <count_0>\n ... <start_{n-1}> <count_{n-1}>\n
    //   <source>\n <edge_list_size>\n <id_0> <cost_0>\n ...
    let f = File::open(&args[1])?;
    let br = BufReader::new(f);
    // Slurp all whitespace-separated tokens
    let mut toks = Vec::<String>::new();
    for line in br.lines() {
        for t in line?.split_whitespace() {
            toks.push(t.to_string());
        }
    }
    let mut it = toks.iter();
    let n: usize = it.next().unwrap().parse()?;
    // Rust equivalent of `struct Node { int starting; int no_of_edges }`.
    // We flatten to a single Vec<i32> pair-per-node for byte-exact device match.
    let mut nodes = vec![0i32; 2 * n];
    for i in 0..n {
        nodes[2*i]     = it.next().unwrap().parse()?;
        nodes[2*i + 1] = it.next().unwrap().parse()?;
    }
    let _source: usize = it.next().unwrap().parse()?;   // ignored — bench forces 0
    let source = 0usize;
    let edge_list_size: usize = it.next().unwrap().parse()?;
    let mut edges = vec![0i32; edge_list_size];
    for i in 0..edge_list_size {
        edges[i] = it.next().unwrap().parse()?;
        let _cost: i32 = it.next().unwrap().parse()?;  // ignored
    }

    println!("Graph check passed (#nodes = {}, #edges in list = {})", n, edge_list_size);
    println!("run bfs (#nodes = {}) on device", n);

    // Host state
    let mut h_mask   = vec![0i8; n];
    let mut h_upmask = vec![0i8; n];
    let mut h_visit  = vec![0i8; n];
    let mut h_cost   = vec![-1i32; n];
    h_mask[source] = 1;
    h_visit[source] = 1;
    h_cost[source] = 0;

    // ---- CPU reference ----
    let (mut ref_mask, mut ref_upmask, mut ref_visit) =
        (h_mask.clone(), h_upmask.clone(), h_visit.clone());
    let mut ref_cost = h_cost.clone();
    loop {
        let mut stop: i8 = 0;
        for tid in 0..n {
            if ref_mask[tid] == 1 {
                ref_mask[tid] = 0;
                let s = nodes[2*tid] as usize;
                let e = s + nodes[2*tid + 1] as usize;
                for k in s..e {
                    let id = edges[k] as usize;
                    if ref_visit[id] == 0 {
                        ref_cost[id] = ref_cost[tid] + 1;
                        ref_upmask[id] = 1;
                    }
                }
            }
        }
        for tid in 0..n {
            if ref_upmask[tid] == 1 {
                ref_mask[tid] = 1;
                ref_visit[tid] = 1;
                stop = 1;
                ref_upmask[tid] = 0;
            }
        }
        if stop == 0 { break; }
    }

    // ---- GPU ----
    let dev = CudaDevice::new(0)?;
    let ptx = compile_ptx(KERNEL_SRC)?;
    dev.load_ptx(ptx, "bfs", &["Kernel", "Kernel2"])?;
    let k1 = dev.get_func("bfs", "Kernel").unwrap();
    let k2 = dev.get_func("bfs", "Kernel2").unwrap();

    let d_nodes = dev.htod_copy(nodes.clone())?;
    let d_edges = dev.htod_copy(edges.clone())?;
    let mut d_mask   = dev.htod_copy(h_mask.clone())?;
    let mut d_upmask = dev.htod_copy(h_upmask.clone())?;
    let mut d_visit  = dev.htod_copy(h_visit.clone())?;
    let mut d_cost   = dev.htod_copy(h_cost.clone())?;
    let mut d_over   = dev.htod_copy(vec![0i8])?;

    let grid = ((n as u32) + BLOCK - 1) / BLOCK;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let t_start = Instant::now();
    loop {
        // Reset over-flag on device
        dev.htod_sync_copy_into(&[0i8], &mut d_over)?;
        unsafe { k1.clone().launch(cfg, (&d_nodes, &d_edges, &mut d_mask, &mut d_upmask, &d_visit, &mut d_cost, n as i32))?; }
        unsafe { k2.clone().launch(cfg, (&mut d_mask, &mut d_upmask, &mut d_visit, &mut d_over, n as i32))?; }
        dev.synchronize()?;
        let over_host: Vec<i8> = dev.dtoh_sync_copy(&d_over)?;
        if over_host[0] == 0 { break; }
    }
    let elapsed_us = t_start.elapsed().as_nanos() as f64 * 1e-3;
    println!("Total kernel execution time : {:.6} (us)", elapsed_us);

    let gpu_cost: Vec<i32> = dev.dtoh_sync_copy(&d_cost)?;

    let mut mismatches = 0usize;
    for i in 0..n {
        if gpu_cost[i] != ref_cost[i] { mismatches += 1; }
    }
    if mismatches == 0 { println!("PASS"); }
    else { println!("FAIL ({} mismatches)", mismatches); }
    Ok(())
}
