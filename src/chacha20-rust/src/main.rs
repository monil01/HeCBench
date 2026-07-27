// ChaCha20 port. The device-side struct methods from chacha20-cuda/chacha20.h
// are inlined verbatim (with __device__ stripped) inside KERNEL_SRC.
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::env;
use std::time::Instant;

const KERNEL_SRC: &str = r#"
typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

struct Chacha20Block {
    uint32_t state[16];

    static __device__ uint32_t rotl32(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }
    static __device__ uint32_t pack4(const uint8_t *a) {
        return uint32_t(a[0] << 0*8) | uint32_t(a[1] << 1*8) |
               uint32_t(a[2] << 2*8) | uint32_t(a[3] << 3*8);
    }
    static __device__ void unpack4(uint32_t src, uint8_t *dst) {
        dst[0] = (src >> 0*8) & 0xff;
        dst[1] = (src >> 1*8) & 0xff;
        dst[2] = (src >> 2*8) & 0xff;
        dst[3] = (src >> 3*8) & 0xff;
    }

    __device__ Chacha20Block(const uint8_t key[32], const uint8_t nonce[8]) {
        const uint8_t *m = (const uint8_t*)"expand 32-byte k";
        state[0]=pack4(m+0); state[1]=pack4(m+4); state[2]=pack4(m+8); state[3]=pack4(m+12);
        state[4]=pack4(key+0);  state[5]=pack4(key+4);  state[6]=pack4(key+8);  state[7]=pack4(key+12);
        state[8]=pack4(key+16); state[9]=pack4(key+20); state[10]=pack4(key+24); state[11]=pack4(key+28);
        state[12]=0; state[13]=0;
        state[14]=pack4(nonce+0); state[15]=pack4(nonce+4);
    }

    __device__ void set_counter(uint64_t c) { state[12] = uint32_t(c); state[13] = uint32_t(c >> 32); }

    __device__ void next32(uint32_t result[16]) {
        for (int i = 0; i < 16; i++) result[i] = state[i];
        #define QR(x,a,b,c,d) \
            x[a]+=x[b]; x[d]=rotl32(x[d]^x[a],16); \
            x[c]+=x[d]; x[b]=rotl32(x[b]^x[c],12); \
            x[a]+=x[b]; x[d]=rotl32(x[d]^x[a], 8); \
            x[c]+=x[d]; x[b]=rotl32(x[b]^x[c], 7);
        for (int i = 0; i < 10; i++) {
            QR(result, 0, 4, 8,12) QR(result, 1, 5, 9,13)
            QR(result, 2, 6,10,14) QR(result, 3, 7,11,15)
            QR(result, 0, 5,10,15) QR(result, 1, 6,11,12)
            QR(result, 2, 7, 8,13) QR(result, 3, 4, 9,14)
        }
        for (int i = 0; i < 16; i++) result[i] += state[i];
        uint32_t *counter = state + 12;
        counter[0]++;
        if (counter[0] == 0) counter[1]++;
    }

    __device__ void next8(uint8_t r8[64]) {
        uint32_t t32[16];
        next32(t32);
        for (int i = 0; i < 16; i++) unpack4(t32[i], r8 + i*4);
    }
};

struct Chacha20 {
    Chacha20Block block;
    uint8_t keystream8[64];
    size_t position;

    __device__ Chacha20(const uint8_t k[32], const uint8_t n[8]): block(k, n), position(64) {
        block.set_counter(0);
    }

    __device__ void crypt(uint8_t *bytes, size_t n_bytes) {
        for (size_t i = 0; i < n_bytes; i++) {
            if (position >= 64) { block.next8(keystream8); position = 0; }
            bytes[i] ^= keystream8[position];
            position++;
        }
    }
};

__device__ void hex_to_raw(const char* src, int n, uint8_t* dst, const uint8_t* c2u) {
    for (int i = threadIdx.x; i < n/2; i += blockDim.x) {
        uint8_t hi = c2u[(unsigned)src[i*2 + 0]];
        uint8_t lo = c2u[(unsigned)src[i*2 + 1]];
        dst[i] = (hi << 4) | lo;
    }
}

extern "C" __global__ void test_keystreams(
    const char* text_key, const char* text_nonce, const char* text_keystream,
    const uint8_t* c2u,
    uint8_t* raw_key, uint8_t* raw_nonce, uint8_t* raw_keystream,
    uint8_t* result,
    int key_sz, int nonce_sz, int ks_sz)
{
    hex_to_raw(text_key, key_sz, raw_key, c2u);
    hex_to_raw(text_nonce, nonce_sz, raw_nonce, c2u);
    hex_to_raw(text_keystream, ks_sz, raw_keystream, c2u);
    __syncthreads();
    if (threadIdx.x == 0) {
        Chacha20 ch(raw_key, raw_nonce);
        ch.crypt(result, ks_sz / 2);
    }
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let repeat: i32 = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(100);

    let mut c2u = [0u8; 256];
    for i in 0..10 { c2u[b'0' as usize + i] = i as u8; }
    for i in 0..26 { c2u[b'a' as usize + i] = (i + 10) as u8; c2u[b'A' as usize + i] = (i + 10) as u8; }

    let h_key = b"000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f";
    let h_nonce = b"0001020304050607";
    let h_keystream: &[u8] = b"f798a189f195e66982105ffb640bb7757f579da31602fc93ec01ac56f85ac3c134a4547b733b46413042c9440049176905d3be59ea1c53f15916155c2be8241a38008b9a26bc35941e2444177c8ade6689de95264986d95889fb60e84629c9bd9a5acb1cc118be563eb9b3a4a472f82e09a7e778492b562ef7130e88dfe031c79db9d4f7c7a899151b9a475032b63fc385245fe054e3dd5a97a5f576fe064025d3ce042c566ab2c507b138db853e3d6959660996546cc9c4a6eafdc777c040d70eaf46f76dad3979e5c5360c3317166a1c894c94a371876a94df7628fe4eaaf2ccb27d5aaae0ad7ad0f9d4b6ad3b54098746d4524d38407a6deb3ab78fab78c9";

    let key_len = h_key.len() as i32;
    let nonce_len = h_nonce.len() as i32;
    let ks_len = h_keystream.len() as i32;
    let result_len = (ks_len / 2) as usize;

    let dev = CudaDevice::new(0)?;
    let ptx = compile_ptx(KERNEL_SRC)?;
    dev.load_ptx(ptx, "chacha", &["test_keystreams"])?;
    let f = dev.get_func("chacha", "test_keystreams").unwrap();

    let d_c2u = dev.htod_copy(c2u.to_vec())?;
    let d_key = dev.htod_copy(h_key.to_vec())?;
    let d_nonce = dev.htod_copy(h_nonce.to_vec())?;
    let d_ks = dev.htod_copy(h_keystream.to_vec())?;
    let mut d_raw_key = dev.alloc_zeros::<u8>((key_len / 2) as usize)?;
    let mut d_raw_nonce = dev.alloc_zeros::<u8>((nonce_len / 2) as usize)?;
    let mut d_raw_ks = dev.alloc_zeros::<u8>(result_len)?;
    let mut d_result = dev.alloc_zeros::<u8>(result_len)?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    dev.synchronize()?;
    let start = Instant::now();
    for _ in 0..repeat {
        dev.htod_copy_into(vec![0u8; result_len], &mut d_result)?;
        unsafe {
            f.clone().launch(cfg, (
                &d_key, &d_nonce, &d_ks, &d_c2u,
                &mut d_raw_key, &mut d_raw_nonce, &mut d_raw_ks, &mut d_result,
                key_len, nonce_len, ks_len,
            ))?;
        }
    }
    dev.synchronize()?;
    let elapsed = start.elapsed();
    println!("Average execution time of kernels: {:.3} (us)",
             elapsed.as_micros() as f64 / repeat as f64);

    let result = dev.dtoh_sync_copy(&d_result)?;
    let raw_ks = dev.dtoh_sync_copy(&d_raw_ks)?;
    let ok = result == raw_ks;
    println!("{}", if ok { "PASS" } else { "FAIL" });
    Ok(())
}
