use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::env;
use std::fs::File;
use std::io::{Read, Write};
use std::time::Instant;

const X_SIZE: usize = 512;
const Y_SIZE: usize = 512;
const PI: f32 = 3.14159265359;
const WHITE: u16 = 1;

const KERNEL_SRC: &str = r#"
#define X_SIZE 512
#define Y_SIZE 512
#define PI     3.14159265359f
#define WHITE  ((unsigned short)1)

extern "C" __global__ void affine(const unsigned short* __restrict__ src,
                                        unsigned short* __restrict__ dst)
{
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;

    const float lx_rot = 30.0f;
    const float ly_rot = 0.0f;
    const float lx_expan = 0.5f;
    const float ly_expan = 0.5f;
    int lx_move = 0;
    int ly_move = 0;
    float affine00, affine01, affine10, affine11;
    float i_affine00, i_affine01, i_affine10, i_affine11;
    float beta0, beta1;
    float i_beta0, i_beta1;
    float det;
    float x_new, y_new;
    float x_frac, y_frac;
    float gray_new;
    int m, n;
    unsigned short output_buffer;

    affine00 = lx_expan * cosf(lx_rot*PI/180.0f);
    affine01 = ly_expan * sinf(ly_rot*PI/180.0f);
    affine10 = lx_expan * sinf(lx_rot*PI/180.0f);
    affine11 = ly_expan * cosf(ly_rot*PI/180.0f);
    beta0 = (float)lx_move;
    beta1 = (float)ly_move;

    det = (affine00 * affine11) - (affine01 * affine10);
    if (det == 0.0f) {
        i_affine00 = 1.0f; i_affine01 = 0.0f;
        i_affine10 = 0.0f; i_affine11 = 1.0f;
        i_beta0 = -beta0; i_beta1 = -beta1;
    } else {
        i_affine00 =  affine11/det;
        i_affine01 = -affine01/det;
        i_affine10 = -affine10/det;
        i_affine11 =  affine00/det;
        i_beta0 = -i_affine00*beta0 - i_affine01*beta1;
        i_beta1 = -i_affine10*beta0 - i_affine11*beta1;
    }

    x_new = i_beta0 + i_affine00*(x - X_SIZE/2.0f) + i_affine01*(y - Y_SIZE/2.0f) + X_SIZE/2.0f;
    y_new = i_beta1 + i_affine10*(x - X_SIZE/2.0f) + i_affine11*(y - Y_SIZE/2.0f) + Y_SIZE/2.0f;

    m = (int)floorf(x_new);
    n = (int)floorf(y_new);
    x_frac = x_new - m;
    y_frac = y_new - n;

    if ((m >= 0) && (m + 1 < X_SIZE) && (n >= 0) && (n+1 < Y_SIZE)) {
        gray_new = (1.0f - y_frac) * ((1.0f - x_frac) * (float)(src[(n * X_SIZE) + m]) +
                                             x_frac  * (float)(src[(n * X_SIZE) + m + 1])) +
                          y_frac  * ((1.0f - x_frac) * (float)(src[((n + 1) * X_SIZE) + m]) +
                                             x_frac  * (float)(src[((n + 1) * X_SIZE) + m + 1]));
        output_buffer = (unsigned short)gray_new;
    } else if (((m + 1 == X_SIZE) && (n >= 0) && (n < Y_SIZE)) ||
               ((n + 1 == Y_SIZE) && (m >= 0) && (m < X_SIZE))) {
        output_buffer = src[(n * X_SIZE) + m];
    } else {
        output_buffer = WHITE;
    }
    dst[(y * X_SIZE) + x] = output_buffer;
}
"#;

fn affine_reference(src: &[u16], dst: &mut [u16]) {
    for y in 0..Y_SIZE {
        for x in 0..X_SIZE {
            let lx_rot = 30.0f32;
            let ly_rot = 0.0f32;
            let lx_expan = 0.5f32;
            let ly_expan = 0.5f32;
            let lx_move = 0i32;
            let ly_move = 0i32;

            let affine00 = lx_expan * (lx_rot * PI / 180.0).cos();
            let affine01 = ly_expan * (ly_rot * PI / 180.0).sin();
            let affine10 = lx_expan * (lx_rot * PI / 180.0).sin();
            let affine11 = ly_expan * (ly_rot * PI / 180.0).cos();
            let beta0 = lx_move as f32;
            let beta1 = ly_move as f32;

            let det = affine00 * affine11 - affine01 * affine10;
            let (i_affine00, i_affine01, i_affine10, i_affine11, i_beta0, i_beta1) = if det == 0.0 {
                (1.0, 0.0, 0.0, 1.0, -beta0, -beta1)
            } else {
                let a00 =  affine11 / det;
                let a01 = -affine01 / det;
                let a10 = -affine10 / det;
                let a11 =  affine00 / det;
                let b0 = -a00 * beta0 - a01 * beta1;
                let b1 = -a10 * beta0 - a11 * beta1;
                (a00, a01, a10, a11, b0, b1)
            };

            let x_new = i_beta0 + i_affine00 * (x as f32 - X_SIZE as f32 / 2.0)
                + i_affine01 * (y as f32 - Y_SIZE as f32 / 2.0)
                + X_SIZE as f32 / 2.0;
            let y_new = i_beta1 + i_affine10 * (x as f32 - X_SIZE as f32 / 2.0)
                + i_affine11 * (y as f32 - Y_SIZE as f32 / 2.0)
                + Y_SIZE as f32 / 2.0;

            let m = x_new.floor() as i32;
            let n = y_new.floor() as i32;
            let x_frac = x_new - m as f32;
            let y_frac = y_new - n as f32;

            let out_val = if m >= 0 && (m + 1) < X_SIZE as i32 && n >= 0 && (n + 1) < Y_SIZE as i32 {
                let m = m as usize; let n = n as usize;
                let gray_new = (1.0 - y_frac) * ((1.0 - x_frac) * src[n * X_SIZE + m] as f32
                    + x_frac * src[n * X_SIZE + m + 1] as f32)
                    + y_frac * ((1.0 - x_frac) * src[(n + 1) * X_SIZE + m] as f32
                        + x_frac * src[(n + 1) * X_SIZE + m + 1] as f32);
                gray_new as u16
            } else if (m + 1 == X_SIZE as i32 && n >= 0 && n < Y_SIZE as i32)
                || (n + 1 == Y_SIZE as i32 && m >= 0 && m < X_SIZE as i32)
            {
                src[n as usize * X_SIZE + m as usize]
            } else {
                WHITE
            };
            dst[y * X_SIZE + x] = out_val;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: {} <input image> <output image> <iterations>", args[0]);
        std::process::exit(1);
    }
    let input_path = &args[1];
    let output_path = &args[2];
    let iterations: i32 = args[3].parse()?;

    let mut buf = Vec::new();
    File::open(input_path)?.read_to_end(&mut buf)?;
    println!("   Reading RAW Image");
    println!("   Bytes read = {}", buf.len());

    assert_eq!(buf.len(), X_SIZE * Y_SIZE * 2, "unexpected input size");
    let mut input_image = vec![0u16; X_SIZE * Y_SIZE];
    for i in 0..X_SIZE * Y_SIZE {
        input_image[i] = u16::from_le_bytes([buf[i * 2], buf[i * 2 + 1]]);
    }

    let dev = CudaDevice::new(0)?;
    let ptx = compile_ptx(KERNEL_SRC)?;
    dev.load_ptx(ptx, "m", &["affine"])?;
    let f = dev.get_func("m", "affine").unwrap();

    let d_in = dev.htod_copy(input_image.clone())?;
    let mut d_out = dev.alloc_zeros::<u16>(X_SIZE * Y_SIZE)?;

    let cfg = LaunchConfig {
        grid_dim: ((X_SIZE / 16) as u32, (Y_SIZE / 16) as u32, 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };

    dev.synchronize()?;
    let start = Instant::now();
    for _ in 0..iterations {
        unsafe { f.clone().launch(cfg, (&d_in, &mut d_out))?; }
    }
    dev.synchronize()?;
    let elapsed = start.elapsed();
    println!("   Average kernel execution time {} (s)", elapsed.as_secs_f64() / iterations as f64);

    let out: Vec<u16> = dev.dtoh_sync_copy(&d_out)?;
    let mut ref_out = vec![0u16; X_SIZE * Y_SIZE];
    affine_reference(&input_image, &mut ref_out);
    let mut max_error = 0i32;
    for i in 0..X_SIZE * Y_SIZE {
        let d = (out[i] as i32 - ref_out[i] as i32).abs();
        if d > max_error { max_error = d; }
    }
    println!("   Max output error is {}", max_error);
    let ok = max_error <= 1;
    println!("{}", if ok { "PASS" } else { "FAIL" });

    // Write result
    let mut fout = File::create(output_path)?;
    for &v in &out {
        fout.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}
