// Haversine port from haversine-cuda. Uses ../haversine-serial/locations.txt
// (2^21 rows of "lat lon"), replicated across 6 reference cities as in the
// CUDA main.
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

const KERNEL_SRC: &str = r#"
#define DEG (3.14159265358979323846 / 180.0)
#define R_KM 6371.0

extern "C" __global__ void compute_haversine_distance(
    const double4* __restrict__ p,
    double* __restrict__ distance,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double ay = p[i].x * DEG;
    double ax = p[i].y * DEG;
    double by = p[i].z * DEG;
    double bx = p[i].w * DEG;
    double x = (bx - ax) / 2.0;
    double y = (by - ay) / 2.0;
    double sinysqrd = sin(y) * sin(y);
    double sinxsqrd = sin(x) * sin(x);
    double scale    = cos(ay) * cos(by);
    distance[i] = 2.0 * R_KM * asin(sqrt(sinysqrd + sinxsqrd * scale));
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let filename = args.get(1).map(String::as_str).unwrap_or("../haversine-serial/locations.txt");
    let repeat: i32 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(100);

    let num_cities = 2_097_152usize;
    let num_ref = 6usize;
    let index_map = [436483usize, 1952407, 627919, 377884, 442703, 1863423];
    let n = num_cities * num_ref;

    println!("Reading city locations from file {}...", filename);
    let file = File::open(filename)?;
    let reader = BufReader::new(file);

    // input packed as (lat, lon, ref_lat, ref_lon) doubles, i.e. 4 doubles/row.
    let mut input = vec![0f64; n * 4];
    let mut idx = 0usize;
    for line in reader.lines() {
        let line = line?;
        let mut it = line.split_ascii_whitespace();
        let lat: f64 = it.next().unwrap().parse()?;
        let lon: f64 = it.next().unwrap().parse()?;
        input[idx * 4 + 0] = lat;
        input[idx * 4 + 1] = lon;
        idx += 1;
        if idx == num_cities { break; }
    }
    // Duplicate for num_ref reference cities.
    for c in 1..num_ref {
        for j in 0..num_cities {
            input[(c * num_cities + j) * 4 + 0] = input[j * 4 + 0];
            input[(c * num_cities + j) * 4 + 1] = input[j * 4 + 1];
        }
    }
    // Fill reference lat/lon slots
    for c in 0..num_ref {
        let ref_idx = index_map[c] - 1;
        let (rlat, rlon) = (input[ref_idx * 4 + 0], input[ref_idx * 4 + 1]);
        for j in c*num_cities..(c+1)*num_cities {
            input[j * 4 + 2] = rlat;
            input[j * 4 + 3] = rlon;
        }
    }

    let dev = CudaDevice::new(0)?;
    let ptx = compile_ptx(KERNEL_SRC)?;
    dev.load_ptx(ptx, "hv", &["compute_haversine_distance"])?;
    let f = dev.get_func("hv", "compute_haversine_distance").unwrap();

    let d_in = dev.htod_copy(input.clone())?;
    let mut d_dist = dev.alloc_zeros::<f64>(n)?;

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
        unsafe { f.clone().launch(cfg, (&d_in, &mut d_dist, n as i32))?; }
    }
    dev.synchronize()?;
    let elapsed = start.elapsed();
    println!("Average kernel execution time {:.6} (s)",
             elapsed.as_secs_f64() / repeat as f64);

    let gpu: Vec<f64> = dev.dtoh_sync_copy(&d_dist)?;

    // CPU reference for verification (single-threaded)
    let mut error: f64 = 0.0;
    let deg = std::f64::consts::PI / 180.0;
    let r_km = 6371.0f64;
    for i in 0..n {
        let ay = input[i*4 + 0] * deg;
        let ax = input[i*4 + 1] * deg;
        let by = input[i*4 + 2] * deg;
        let bx = input[i*4 + 3] * deg;
        let x = (bx - ax) / 2.0;
        let y = (by - ay) / 2.0;
        let siny = y.sin(); let sinx = x.sin();
        let d = 2.0 * r_km * (siny*siny + sinx*sinx * ay.cos() * by.cos()).sqrt().asin();
        let e = (gpu[i] - d).abs();
        if e > error { error = e; }
    }
    println!("The maximum error in distance is {}", error);
    println!("{}", if error < 1e-6 { "PASS" } else { "FAIL" });
    Ok(())
}
