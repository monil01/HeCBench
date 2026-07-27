use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::time::Instant;

// KERNEL_SRC defines a family of typed copy kernels. Each kernel copies numElements
// TData objects from d_idata to d_odata. We test aligned vs misaligned structs.

const KERNEL_SRC: &str = r#"
typedef unsigned char uchar_misaligned;
typedef unsigned short int ushort_misaligned;

typedef struct { unsigned char r, g, b, a; } uchar4_misaligned;
typedef struct { unsigned int l, a; } uint2_misaligned;
typedef struct { unsigned int r, g, b; } uint3_misaligned;
typedef struct { unsigned int r, g, b, a; } uint4_misaligned;
typedef struct { uint4_misaligned c1, c2; } uint8_misaligned;

typedef struct __align__(4) { unsigned char r, g, b, a; } uchar4_aligned;
typedef unsigned int uint_aligned;
typedef struct __align__(8) { unsigned int l, a; } uint2_aligned;
typedef struct __align__(16) { unsigned int r, g, b; } uint3_aligned;
typedef struct __align__(16) { unsigned int r, g, b, a; } uint4_aligned;
typedef struct __align__(16) { uint4_aligned c1, c2; } uint8_aligned;

#define COPY_KERNEL(NAME, T) \
extern "C" __global__ void NAME(T* __restrict__ d_odata, const T* __restrict__ d_idata, int n) { \
    const int pos = blockDim.x * blockIdx.x + threadIdx.x; \
    if (pos < n) d_odata[pos] = d_idata[pos]; \
}

COPY_KERNEL(copy_uchar_misaligned, uchar_misaligned)
COPY_KERNEL(copy_uchar4_misaligned, uchar4_misaligned)
COPY_KERNEL(copy_uchar4_aligned, uchar4_aligned)
COPY_KERNEL(copy_ushort_misaligned, ushort_misaligned)
COPY_KERNEL(copy_uint_aligned, uint_aligned)
COPY_KERNEL(copy_uint2_misaligned, uint2_misaligned)
COPY_KERNEL(copy_uint2_aligned, uint2_aligned)
COPY_KERNEL(copy_uint3_misaligned, uint3_misaligned)
COPY_KERNEL(copy_uint3_aligned, uint3_aligned)
COPY_KERNEL(copy_uint4_misaligned, uint4_misaligned)
COPY_KERNEL(copy_uint4_aligned, uint4_aligned)
COPY_KERNEL(copy_uint8_misaligned, uint8_misaligned)
COPY_KERNEL(copy_uint8_aligned, uint8_aligned)
"#;

const MEM_SIZE_RAW: usize = 50_000_000;
const NUM_ITERATIONS: i32 = 1000;

fn run_test(
    name: &str,
    dev: &std::sync::Arc<CudaDevice>,
    d_in: &cudarc::driver::CudaSlice<u8>,
    d_out: &mut cudarc::driver::CudaSlice<u8>,
    kernel_name: &str,
    element_size: usize,       // sizeof(TData) as compiled by nvrtc (incl padding)
    packed_size: usize,        // meaningful bytes
    mem_size: usize,
    h_in: &[u8],
) -> Result<i32, Box<dyn std::error::Error>> {
    let num_elements = mem_size / element_size;
    let total_mem_aligned = num_elements * element_size;

    dev.memset_zeros(d_out)?;
    dev.synchronize()?;

    let cfg = LaunchConfig {
        grid_dim: (((num_elements as u32) + 255) / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let f = dev.get_func("m", kernel_name).unwrap();

    let start = Instant::now();
    for _ in 0..NUM_ITERATIONS {
        unsafe { f.clone().launch(cfg, (&mut *d_out, d_in as &_, num_elements as i32))?; }
    }
    dev.synchronize()?;
    let elapsed = start.elapsed();
    let gpu_time = elapsed.as_secs_f64() / NUM_ITERATIONS as f64;
    println!(
        "Avg. time: {:.6} ms / Copy throughput: {:.6} GB/s.",
        gpu_time * 1000.0,
        total_mem_aligned as f64 / (gpu_time * 1073741824.0)
    );

    let out_bytes: Vec<u8> = dev.dtoh_sync_copy(d_out)?;

    // Compare packed_size bytes per element
    let mut ok = 1;
    for pos in 0..num_elements {
        let base = pos * element_size;
        for i in 0..packed_size {
            if h_in[base + i] != out_bytes[base + i] { ok = 0; break; }
        }
        if ok == 0 { break; }
    }
    println!("\t{}: TEST {}", name, if ok == 1 { "PASS" } else { "FAIL" });
    Ok(if ok == 1 { 0 } else { 1 })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let memory_size = MEM_SIZE_RAW & 0xffffff00;
    let mut h_in = vec![0u8; memory_size];
    for i in 0..memory_size { h_in[i] = ((i & 0xFF) + 1) as u8; }

    let dev = CudaDevice::new(0)?;
    let ptx = compile_ptx(KERNEL_SRC)?;
    let kernel_names = [
        "copy_uchar_misaligned",
        "copy_uchar4_misaligned",
        "copy_uchar4_aligned",
        "copy_ushort_misaligned",
        "copy_uint_aligned",
        "copy_uint2_misaligned",
        "copy_uint2_aligned",
        "copy_uint3_misaligned",
        "copy_uint3_aligned",
        "copy_uint4_misaligned",
        "copy_uint4_aligned",
        "copy_uint8_misaligned",
        "copy_uint8_aligned",
    ];
    dev.load_ptx(ptx, "m", &kernel_names)?;

    let d_in = dev.htod_copy(h_in.clone())?;
    let mut d_out = dev.alloc_zeros::<u8>(memory_size)?;

    let mut total_failures = 0;
    // (name, kernel, elem_size, packed_size)
    // element sizes:
    //   uchar_misaligned = 1, uchar4_misaligned = 4 (natural), uchar4_aligned = 4
    //   ushort_misaligned = 2
    //   uint_aligned = 4
    //   uint2_misaligned = 8 (natural), uint2_aligned = 8
    //   uint3_misaligned = 12 (natural align 4), uint3_aligned = 16 (align 16 -> pads to 16)
    //   uint4_misaligned = 16, uint4_aligned = 16
    //   uint8_misaligned = 32, uint8_aligned = 32
    let tests: &[(&str, &str, usize, usize)] = &[
        ("uchar_misaligned", "copy_uchar_misaligned", 1, 1),
        ("uchar4_misaligned", "copy_uchar4_misaligned", 4, 4),
        ("uchar4_aligned", "copy_uchar4_aligned", 4, 4),
        ("ushort_misaligned", "copy_ushort_misaligned", 2, 2),
        ("uint_aligned", "copy_uint_aligned", 4, 4),
        ("uint2_misaligned", "copy_uint2_misaligned", 8, 8),
        ("uint2_aligned", "copy_uint2_aligned", 8, 8),
        ("uint3_misaligned", "copy_uint3_misaligned", 12, 12),
        ("uint3_aligned", "copy_uint3_aligned", 16, 12),
        ("uint4_misaligned", "copy_uint4_misaligned", 16, 16),
        ("uint4_aligned", "copy_uint4_aligned", 16, 16),
        ("uint8_misaligned", "copy_uint8_misaligned", 32, 32),
        ("uint8_aligned", "copy_uint8_aligned", 32, 32),
    ];
    for &(name, kname, esize, psize) in tests {
        println!("{}...", name);
        total_failures += run_test(name, &dev, &d_in, &mut d_out, kname, esize, psize, memory_size, &h_in)?;
    }
    println!("\n[alignedTypes] -> Test Results: {} Failures", total_failures);
    if total_failures == 0 {
        println!("PASS");
    } else {
        println!("FAIL");
    }
    Ok(())
}
