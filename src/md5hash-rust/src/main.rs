// Rust port of the `md5hash` HeCBench benchmark (simplified variant,
// matches md5hash-triton's approach: byteLength=7, valsPerByte=10,
// 10M keys per pass).
//
// The GPU kernel is the CUDA source's `md5_2words` + `md5hash_kernel`
// lifted verbatim as an inline NVRTC source string. Correctness: we
// hash a target key with the Rust `md-5` crate, launch the kernel,
// and require it to find the same index+digest.
//
// Usage: md5hash-rust <offload> <passes>   (offload arg kept for CLI
//                                            parity with the CUDA
//                                            benchmark; unused)
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use md5::{Digest, Md5};
use std::env;
use std::time::Instant;

const BYTE_LEN: i32 = 7;
const VALS_PER_BYTE: i32 = 10;
const BLOCK: u32 = 256;

const KERNEL_SRC: &str = r##"
// leftrotate
#define LEFTROTATE(x, c) (((x) << (c)) | ((x) >> (32 - (c))))
#define F(x,y,z) ((x & y) | ((~x) & z))
#define G(x,y,z) ((x & z) | ((~z) & y))
#define H(x,y,z) (x ^ y ^ z)
#define I(x,y,z) (y ^ (x | (~z)))
#define ROUND(w, r, k, v, x, y, z, func) \
{ \
    a = a + func(b,c,d) + k + w; \
    unsigned int temp = d; \
    d = c; \
    c = b; \
    b = b + LEFTROTATE(a, r); \
    a = temp; \
}

__device__ inline void md5_2words(unsigned int *words, unsigned int len,
                                  unsigned int *digest)
{
    unsigned int h0 = 0x67452301u;
    unsigned int h1 = 0xefcdab89u;
    unsigned int h2 = 0x98badcfeu;
    unsigned int h3 = 0x10325476u;
    unsigned int a = h0, b = h1, c = h2, d = h3;
    unsigned int WL = len * 8u;
    unsigned int W0 = words[0];
    unsigned int W1 = words[1];
    switch (len) {
      case 0: W0 |= 0x00000080u; break;
      case 1: W0 |= 0x00008000u; break;
      case 2: W0 |= 0x00800000u; break;
      case 3: W0 |= 0x80000000u; break;
      case 4: W1 |= 0x00000080u; break;
      case 5: W1 |= 0x00008000u; break;
      case 6: W1 |= 0x00800000u; break;
      case 7: W1 |= 0x80000000u; break;
    }
    ROUND(W0,  7, 0xd76aa478u, a,b,c,d, F);
    ROUND(W1, 12, 0xe8c7b756u, d,a,b,c, F);
    ROUND(0,  17, 0x242070dbu, c,d,a,b, F);
    ROUND(0,  22, 0xc1bdceeeu, b,c,d,a, F);
    ROUND(0,   7, 0xf57c0fafu, a,b,c,d, F);
    ROUND(0,  12, 0x4787c62au, d,a,b,c, F);
    ROUND(0,  17, 0xa8304613u, c,d,a,b, F);
    ROUND(0,  22, 0xfd469501u, b,c,d,a, F);
    ROUND(0,   7, 0x698098d8u, a,b,c,d, F);
    ROUND(0,  12, 0x8b44f7afu, d,a,b,c, F);
    ROUND(0,  17, 0xffff5bb1u, c,d,a,b, F);
    ROUND(0,  22, 0x895cd7beu, b,c,d,a, F);
    ROUND(0,   7, 0x6b901122u, a,b,c,d, F);
    ROUND(0,  12, 0xfd987193u, d,a,b,c, F);
    ROUND(WL, 17, 0xa679438eu, c,d,a,b, F);
    ROUND(0,  22, 0x49b40821u, b,c,d,a, F);

    ROUND(W1,  5, 0xf61e2562u, a,b,c,d, G);
    ROUND(0,   9, 0xc040b340u, d,a,b,c, G);
    ROUND(0,  14, 0x265e5a51u, c,d,a,b, G);
    ROUND(W0, 20, 0xe9b6c7aau, b,c,d,a, G);
    ROUND(0,   5, 0xd62f105du, a,b,c,d, G);
    ROUND(0,   9, 0x02441453u, d,a,b,c, G);
    ROUND(0,  14, 0xd8a1e681u, c,d,a,b, G);
    ROUND(0,  20, 0xe7d3fbc8u, b,c,d,a, G);
    ROUND(0,   5, 0x21e1cde6u, a,b,c,d, G);
    ROUND(WL,  9, 0xc33707d6u, d,a,b,c, G);
    ROUND(0,  14, 0xf4d50d87u, c,d,a,b, G);
    ROUND(0,  20, 0x455a14edu, b,c,d,a, G);
    ROUND(0,   5, 0xa9e3e905u, a,b,c,d, G);
    ROUND(0,   9, 0xfcefa3f8u, d,a,b,c, G);
    ROUND(0,  14, 0x676f02d9u, c,d,a,b, G);
    ROUND(0,  20, 0x8d2a4c8au, b,c,d,a, G);

    ROUND(0,   4, 0xfffa3942u, a,b,c,d, H);
    ROUND(0,  11, 0x8771f681u, d,a,b,c, H);
    ROUND(0,  16, 0x6d9d6122u, c,d,a,b, H);
    ROUND(WL, 23, 0xfde5380cu, b,c,d,a, H);
    ROUND(W1,  4, 0xa4beea44u, a,b,c,d, H);
    ROUND(0,  11, 0x4bdecfa9u, d,a,b,c, H);
    ROUND(0,  16, 0xf6bb4b60u, c,d,a,b, H);
    ROUND(0,  23, 0xbebfbc70u, b,c,d,a, H);
    ROUND(0,   4, 0x289b7ec6u, a,b,c,d, H);
    ROUND(W0, 11, 0xeaa127fau, d,a,b,c, H);
    ROUND(0,  16, 0xd4ef3085u, c,d,a,b, H);
    ROUND(0,  23, 0x04881d05u, b,c,d,a, H);
    ROUND(0,   4, 0xd9d4d039u, a,b,c,d, H);
    ROUND(0,  11, 0xe6db99e5u, d,a,b,c, H);
    ROUND(0,  16, 0x1fa27cf8u, c,d,a,b, H);
    ROUND(0,  23, 0xc4ac5665u, b,c,d,a, H);

    ROUND(W0,  6, 0xf4292244u, a,b,c,d, I);
    ROUND(0,  10, 0x432aff97u, d,a,b,c, I);
    ROUND(WL, 15, 0xab9423a7u, c,d,a,b, I);
    ROUND(0,  21, 0xfc93a039u, b,c,d,a, I);
    ROUND(0,   6, 0x655b59c3u, a,b,c,d, I);
    ROUND(0,  10, 0x8f0ccc92u, d,a,b,c, I);
    ROUND(0,  15, 0xffeff47du, c,d,a,b, I);
    ROUND(W1, 21, 0x85845dd1u, b,c,d,a, I);
    ROUND(0,   6, 0x6fa87e4fu, a,b,c,d, I);
    ROUND(0,  10, 0xfe2ce6e0u, d,a,b,c, I);
    ROUND(0,  15, 0xa3014314u, c,d,a,b, I);
    ROUND(0,  21, 0x4e0811a1u, b,c,d,a, I);
    ROUND(0,   6, 0xf7537e82u, a,b,c,d, I);
    ROUND(0,  10, 0xbd3af235u, d,a,b,c, I);
    ROUND(0,  15, 0x2ad7d2bbu, c,d,a,b, I);
    ROUND(0,  21, 0xeb86d391u, b,c,d,a, I);

    digest[0] = h0 + a;
    digest[1] = h1 + b;
    digest[2] = h2 + c;
    digest[3] = h3 + d;
}

__device__ inline void IndexToKey(unsigned int index, int byteLength,
                                  int valsPerByte, unsigned char vals[8])
{
    for (int i=0; i<8; ++i) vals[i]=0;
    for (int i=0; i<byteLength; ++i) {
        vals[i] = index % valsPerByte;
        index /= valsPerByte;
    }
}

extern "C" __global__ void
md5hash_kernel(int* foundIndex, unsigned char* foundKey, unsigned int* foundDigest,
               int keyspace, int byteLength, int valsPerByte,
               unsigned int sd0, unsigned int sd1, unsigned int sd2, unsigned int sd3)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int startindex = tid * valsPerByte;
    unsigned char key[8] = {0,0,0,0,0,0,0,0};
    IndexToKey(startindex, byteLength, valsPerByte, key);
    for (int j=0; j<valsPerByte && startindex+j<keyspace; ++j) {
        unsigned int digest[4];
        md5_2words((unsigned int*)key, byteLength, digest);
        if (digest[0]==sd0 && digest[1]==sd1 && digest[2]==sd2 && digest[3]==sd3) {
            *foundIndex = startindex + j;
            for (int k=0; k<8; ++k) foundKey[k] = key[k];
            foundDigest[0]=digest[0]; foundDigest[1]=digest[1];
            foundDigest[2]=digest[2]; foundDigest[3]=digest[3];
        }
        ++key[0];
    }
}
"##;

fn keyspace_size(byte_len: i32, vals_per_byte: i32) -> i32 {
    let mut k: i32 = 1;
    for _ in 0..byte_len {
        k = k.saturating_mul(vals_per_byte);
    }
    k
}

fn index_to_key(mut index: i32, byte_len: i32, vals_per_byte: i32) -> [u8; 8] {
    let mut out = [0u8; 8];
    for i in 0..byte_len as usize {
        out[i] = (index % vals_per_byte) as u8;
        index /= vals_per_byte;
    }
    out
}

fn md5_of(key: &[u8; 8]) -> [u32; 4] {
    let mut hasher = Md5::new();
    hasher.update(&key[..BYTE_LEN as usize]);
    let d = hasher.finalize();
    let mut out = [0u32; 4];
    for i in 0..4 {
        out[i] = u32::from_le_bytes([d[4*i], d[4*i+1], d[4*i+2], d[4*i+3]]);
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    // Args are kept for CLI parity with the CUDA benchmark: <offload> <passes>.
    // Both are ignored; we sweep the fixed byteLength=7, valsPerByte=10 space.
    let passes: usize = if args.len() > 2 { args[2].parse().unwrap_or(1) } else { 1 };

    let keyspace = keyspace_size(BYTE_LEN, VALS_PER_BYTE);
    println!("MD5 keyspace: {} (byteLength=7, valsPerByte=10)", keyspace);

    // Pick a target index deterministically, hash it, then let the GPU find it.
    let target_index: i32 = keyspace / 3 + 42;
    let target_key = index_to_key(target_index, BYTE_LEN, VALS_PER_BYTE);
    let target_digest = md5_of(&target_key);
    println!(
        "Target index {}, digest {:08x}{:08x}{:08x}{:08x}",
        target_index,
        target_digest[0].swap_bytes(),
        target_digest[1].swap_bytes(),
        target_digest[2].swap_bytes(),
        target_digest[3].swap_bytes()
    );

    let dev = CudaDevice::new(0)?;
    let ptx = compile_ptx(KERNEL_SRC)?;
    dev.load_ptx(ptx, "md5", &["md5hash_kernel"])?;
    let f = dev.get_func("md5", "md5hash_kernel").unwrap();

    let mut d_found_idx = dev.htod_copy(vec![-1i32])?;
    let mut d_found_key = dev.htod_copy(vec![0u8; 8])?;
    let mut d_found_dig = dev.htod_copy(vec![0u32; 4])?;

    // Kernel processes valsPerByte keys per thread → grid = keyspace/valsPerByte/BLOCK
    let n_threads = ((keyspace + VALS_PER_BYTE - 1) / VALS_PER_BYTE) as u32;
    let grid = (n_threads + BLOCK - 1) / BLOCK;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    dev.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..passes {
        unsafe {
            f.clone().launch(cfg, (
                &mut d_found_idx, &mut d_found_key, &mut d_found_dig,
                keyspace, BYTE_LEN, VALS_PER_BYTE,
                target_digest[0], target_digest[1], target_digest[2], target_digest[3],
            ))?;
        }
    }
    dev.synchronize()?;
    let elapsed = t0.elapsed();
    let hashes = (keyspace as u64) * (passes as u64);
    let mhps = hashes as f64 / elapsed.as_secs_f64() / 1e6;
    println!("Passes: {}, keyspace {} => {} hashes in {:.2} s = {:.2} M/s",
             passes, keyspace, hashes, elapsed.as_secs_f64(), mhps);

    let found_idx: Vec<i32> = dev.dtoh_sync_copy(&d_found_idx)?;
    let found_key: Vec<u8> = dev.dtoh_sync_copy(&d_found_key)?;
    let found_dig: Vec<u32> = dev.dtoh_sync_copy(&d_found_dig)?;
    println!("Found: index={}, key={:?}, digest[0]={:08x}", found_idx[0], &found_key[..7], found_dig[0]);

    let ok = found_idx[0] == target_index
        && found_key[..7] == target_key[..7]
        && found_dig[..4] == target_digest[..4];
    println!("{}", if ok { "PASS" } else { "FAIL" });
    Ok(())
}
