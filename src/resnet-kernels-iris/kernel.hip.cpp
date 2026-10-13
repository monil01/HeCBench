#include <hip/hip_runtime.h>


const char inputName128one[] = "data/input_one_14_1024.bin";
const char weightName128one[] = "data/weight_one_1024.bin";
const char bnBias_myKernel_Name128one[] = "data/bnBias_myKernel_one_1024.bin";
const char bnScale_myKernel_Name128one[] = "data/bnScale_myKernel_one_1024.bin";

extern "C" __global__ void kernel_512_one_128(
  const float *__restrict__ A,
  const float *__restrict__ B,
  const float *__restrict__ bnBias,
  const float *__restrict__ bnScale,
        float *__restrict__ C) 
{
  int tile = blockIdx.x, in_channel = threadIdx.x, line = threadIdx.y;
  int ind = line*128 + in_channel;

  __shared__ float shared_[4*512 + 64*128 + 4*128 + 2*128];
  float *__restrict__ weights = shared_ + 512*4,
        *__restrict__ output = weights + 128*64,
        *__restrict__ input = shared_;
  float *__restrict__ bias = output + 4*128,
        *__restrict__ scale = bias + 128;

  for (int i = 0; i < 4; i++)
    input[ind + i*512] = A[tile*2048 + i*512 + ind];
  bias[in_channel] = bnBias[in_channel];
  scale[in_channel] = bnScale[in_channel];
  output[ind] = 0.0f;
  __syncthreads();

  for (int k = 0; k < 512; k += 64) {
    const float *B_start = B + k*128;
    for (int i = 0; i < 16; i++)
      weights[ind + i*512] = B_start[i*512 + ind];
    __syncthreads();

    const float *A_start = input + k;
    for (int p = 0; p < 64; p++) {
      output[ind] += A_start[line*512 + p] * weights[in_channel + p*128];
    }
    __syncthreads();
  }

  float *C_start = C + tile*512, res = scale[in_channel] * output[ind] + bias[in_channel];
  C_start[ind] = res > 0 ? res : 0;
}

extern "C" __global__ void kernel_128_one_512(
  const float *__restrict__ A,
  const float *__restrict__ B,
  const float *__restrict__ bnBias,
  const float *__restrict__ bnScale,
        float *__restrict__ C) 
{
  int tile = blockIdx.x, part = blockIdx.y, in_channel = threadIdx.x, line = threadIdx.y;
  int ind = line*128 + in_channel;

  __shared__ float shared_[4*128 + 64*128 + 4*128 + 2*128];
  float *weights = shared_ + 128*4, *output = weights + 128*64, *input = shared_;
  float *bias = output + 4*128, *scale = bias + 128;

  input[ind] = A[tile * 512 + ind];
  bias[in_channel] = bnBias[part*128 + in_channel];
  scale[in_channel] = bnScale[part*128+ in_channel];
  output[ind] = 0.0f;
  __syncthreads();

  for (int k = 0; k < 128; k += 64) {
    for (int i = 0; i < 16; i++)
      weights[ind + 512*i] = B[(k + i*4 + line)*512 + part*128 + in_channel];
    __syncthreads();

    float *A_start = input + k;
    for (int p = 0; p < 64; p++) {
      output[ind] += A_start[line*128 + p] * weights[in_channel + p*128];
    }
    __syncthreads();
  }

  float *C_start = C + tile*2048 + part*128;
  float res = scale[in_channel] * output[ind] + bias[in_channel];
  C_start[line * 512 + in_channel] = res;
}



const char inputName256one[] = "data/input_one_14_1024.bin";
const char weightName256one[] = "data/weight_one_1024.bin";
const char bnBias_myKernel_Name256one[] = "data/bnBias_myKernel_one_1024.bin";
const char bnScale_myKernel_Name256one[] = "data/bnScale_myKernel_one_1024.bin";

extern "C" __global__ void kernel_1024_one_256(
  const float *__restrict__ A,
  const float *__restrict__ B,
  const float *__restrict__ bnBias,
  const float *__restrict__ bnScale,
        float *__restrict__ C) 
{
  int tile = blockIdx.x, in_channel = threadIdx.x, line = threadIdx.y;
  int ind = line*256 + in_channel;

  __shared__ float shared_[4*1024 + 16*256 + 4*256 + 2*256];
  float *__restrict__ weights = shared_ + 1024*4,
        *__restrict__ output = weights + 256*16,
        *__restrict__ input = shared_;
  float *__restrict__ bias = output + 4*256,
        *__restrict__ scale = bias + 256;

  for (int i = 0; i < 4; i++)
    input[ind + i*1024] = A[tile*4096 + i*1024 + ind];
  bias[in_channel] = bnBias[in_channel];
  scale[in_channel] = bnScale[in_channel];
  output[ind] = 0.0f;
  __syncthreads();

  for (int k = 0; k < 1024; k += 16) {
    const float *B_start = B + k*256;
    for (int i = 0; i < 4; i++)
      weights[ind + i*1024] = B_start[i*1024 + ind];
    __syncthreads();

    const float *A_start = input + k;
    for (int p = 0; p < 16; p++) {
      output[ind] += A_start[line*1024 + p] * weights[in_channel + p*256];
    }
    __syncthreads();
  }

  float *C_start = C + tile*1024, res = scale[in_channel] * output[ind] + bias[in_channel];
  C_start[ind] = res > 0 ? res : 0;
}

extern "C" __global__ void kernel_256_one_1024(
  const float *__restrict__ A,
  const float *__restrict__ B,
  const float *__restrict__ bnBias,
  const float *__restrict__ bnScale,
        float *__restrict__ C) 
{
  int tile = blockIdx.x, part = blockIdx.y, in_channel = threadIdx.x, line = threadIdx.y;
  int ind = line*256 + in_channel;

  __shared__ float shared_[4*256 + 32*256 + 4*256 + 2*256];
  float *weights = shared_ + 256*4, *output = weights + 256*32, *input = shared_;
  float *bias = output + 4*256, *scale = bias + 256;

  input[ind] = A[tile * 1024 + ind];
  bias[in_channel] = bnBias[part*256 + in_channel];
  scale[in_channel] = bnScale[part*256+ in_channel];
  output[ind] = 0.0f;
  __syncthreads();

  for (int k = 0; k < 256; k += 32) {
    for (int i = 0; i < 8; i++)
      weights[ind + 1024*i] = B[(k + i*4 + line)*1024 + part*256 + in_channel];
    __syncthreads();

    float *A_start = input + k;
    for (int p = 0; p < 32; p++) {
      output[ind] += A_start[line*256 + p] * weights[in_channel + p*256];
    }
    __syncthreads();
  }

  float *C_start = C + tile*4096 + part*256;
  C_start[line * 1024 + in_channel] = scale[in_channel] * output[ind] + bias[in_channel];
}



const char inputName128[] = "data/input_14_1_128.bin";
const char biasName128[] = "data/bias_128.bin";
const char weight_winograd_Name128[] = "data/weight_winograd_128_128.bin";
const char bnBias_winograd_Name128[] = "data/bnBias_winograd_128.bin";
const char bnScale_winograd_Name128[] = "data/bnScale_winograd_128.bin";

#define d(input, i, j, Inz) ( input[Inz + i*768 + (j<<7)] )

extern "C" __global__ void kernel_128_winograd_BtdB(
  const float *__restrict__ pInputs,
        float *__restrict__ pOutputs)
{
  int Inx = blockIdx.x<<2, Iny0 = blockIdx.y<<2, Iny1 = threadIdx.y, Inz = threadIdx.x;
  int Iny = Iny0+Iny1, stride_r = 2048, stride_c = 128; // 2048 = 16*128
  int c_glb_start = Inx*stride_r + Iny*stride_c + Inz, c_input = Iny1*stride_c + Inz;

  __shared__ float input[6*6*128];

  int tmp[6] = {0, 768, 1536, 2304, 3072, 3840}; // 768 = 6*128
  for (int i = 0; i < 6; i++) {
    input[c_input + tmp[i]] = pInputs[c_glb_start + i*stride_r];
  }
  __syncthreads();

  float BTd[6];
  switch(Iny1) {
    case 0:
      for (int j = 0; j < 6; j++) {
        BTd[j] = d(input, 0, j, Inz)*4 - d(input, 2, j, Inz)*5 + d(input, 4, j, Inz);
      }
      break;
    case 1:
      for (int j = 0; j < 6; j++) {
        BTd[j] = -d(input, 1, j, Inz)*4 - d(input, 2, j, Inz)*4 + d(input, 3, j, Inz) + d(input, 4, j, Inz);
      }
      break;
    case 2:
      for (int j = 0; j < 6; j++) {
        BTd[j] = d(input, 1, j, Inz)*4 - d(input, 2, j, Inz)*4 - d(input, 3, j, Inz) + d(input, 4, j, Inz);
      }
      break;
    case 3:
      for (int j = 0; j < 6; j++) {
        BTd[j] = -d(input, 1, j, Inz)*2 - d(input, 2, j, Inz) + d(input, 3, j, Inz)*2 + d(input, 4, j, Inz);
      }
      break;
    case 4:
      for (int j = 0; j < 6; j++) {
        BTd[j] = d(input, 1, j, Inz)*2 - d(input, 2, j, Inz) - d(input, 3, j, Inz)*2 + d(input, 4, j, Inz);
      }
      break;
    case 5:
      for (int j = 0; j < 6; j++) {
        BTd[j] = d(input, 1, j, Inz)*4 - d(input, 3, j, Inz)*5 + d(input, 5, j, Inz);
      }
      break;
  }
  __syncthreads();

  int tmp_offset = Iny1*768+Inz;
  for (int i = 0; i < 6; i++) {
    input[tmp_offset + i*stride_c] = BTd[i];
  }
  __syncthreads();

  float BTdB[6];
  switch(Iny1) {
    case 0:
      for (int i = 0; i < 6; i++) {
        BTdB[i] = 4*d(input, i, 0, Inz) - 5*d(input, i, 2, Inz) + d(input, i, 4, Inz);
      }
      break;
    case 1:
      for (int i = 0; i < 6; i++) {
        BTdB[i] = -4*d(input, i, 1, Inz) - 4*d(input, i, 2, Inz) + d(input, i, 3, Inz) + d(input, i, 4, Inz);
      }
      break;
    case 2:
      for (int i = 0; i < 6; i++) {
        BTdB[i] = 4*d(input, i, 1, Inz) - 4*d(input, i, 2, Inz) - d(input, i, 3, Inz) + d(input, i, 4, Inz);
      }
      break;
    case 3:
      for (int i = 0; i < 6; i++) {
        BTdB[i] = -2*d(input, i, 1, Inz) - d(input, i, 2, Inz) + 2*d(input, i, 3, Inz) + d(input, i, 4, Inz);
      }
      break;
    case 4:
      for (int i = 0; i < 6; i++) {
        BTdB[i] = 2*d(input, i, 1, Inz) - d(input, i, 2, Inz) - 2*d(input, i, 3, Inz) + d(input, i, 4, Inz);
      }
      break;
    case 5:
      for (int i = 0; i < 6; i++) {
        BTdB[i] = 4*d(input, i, 1, Inz) - 5*d(input, i, 3, Inz) + d(input, i, 5, Inz);
      }
      break;
  }
  __syncthreads();

  for (int i = 0; i < 6; i++) {
    pOutputs[(Iny1 + i*6)*2048 + (blockIdx.x*4+blockIdx.y)*128 + Inz] = BTdB[i];
  }
}

extern "C" __global__ void kernel_128_winograd_AtIA(
  const float *__restrict__ pInputs,
  const float *__restrict__ pBiases,
  const float *__restrict__ pScales,
        float *__restrict__ pOutputs)
{
  int Tilex = blockIdx.x, Tiley = blockIdx.y, Iny = threadIdx.y, kz = blockIdx.z, Inx = threadIdx.x;
  int c_input = Inx*6 + Iny;

  __shared__ float bias, scale;
  __shared__ float input[6*6];

  input[c_input] = pInputs[c_input*16*128 + (Tilex*4+Tiley)*128 + kz];
  bias = pBiases[kz];
  scale = pScales[kz];
  __syncthreads();

  float tmp = 0;
  switch(Inx) {
    case 0:
      tmp = input[Iny] + input[6+Iny] + input[12+Iny] + input[18+Iny] + input[24+Iny];
      break;
    case 1:
      tmp = input[6+Iny] - input[12+Iny] + 2*input[18+Iny] - 2*input[24+Iny];
      break;
    case 2:
      tmp = input[6+Iny] + input[12+Iny] + 4*input[18+Iny] + 4*input[24+Iny];
      break;
    case 3:
      tmp = input[6+Iny] - input[12+Iny] + 8*input[18+Iny] - 8*input[24+Iny] + input[30+Iny];
      break;
  }
  __syncthreads();

  input[c_input] = tmp;
  __syncthreads();

  if (Inx > 3 || (Tilex == 3 && Inx > 1)) return;

  int x;
  float o;
  switch(Iny) {
    case 0:
      x = Inx*6;
      o = scale*(input[x]+input[x+1]+input[x+2]+input[x+3]+input[x+4])+ bias;
      pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+1)*128 + kz] = o > 0 ? o : 0;
      break;
    case 1:
      x = Inx*6;
      o = scale*(input[x+1] - input[x+2] + 2*input[x+3] - 2*input[x+4]) + bias;
      pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+2)*128 + kz] = o > 0 ? o : 0;
      break;
    case 2:
      if (Tiley == 3) break;
      x = Inx*6;
      o = scale*(input[x+1] + input[x+2] + 4*input[x+3] + 4*input[x+4]) + bias;
      pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+3)*128 + kz] = o > 0 ? o : 0;
      break;
    case 3:
      if (Tiley == 3) break;
      x = Inx*6;
      o = scale*(input[x+1] - input[x+2] + 8*input[x+3] - 8*input[x+4] + input[x+5]) + bias;
      pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+4)*128 + kz] = o > 0 ? o : 0;
      break;
  }
}

extern "C" __global__ void kernel_128_OuterProduct_128(
  const float *__restrict__ A,
  const float *__restrict__ B,
        float *__restrict__ C)
{
  int Tile = blockIdx.x, Part = blockIdx.y, tX = threadIdx.x, tY = threadIdx.y;
  int c_input = tY*128 + tX, c_kernel = c_input;
  int T_offset = (Tile<<11) + (Part<<10) + c_input;
  int B_offset = (Tile<<14) + c_kernel;

  __shared__ float input[8*128 + 64*128 + 8*128];
  float *kernel = input + 1024, *out = kernel + 8192;
  int B_stride[32] = {0, 128, 256, 384, 512, 640, 768, 896,
                      1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920,
                      2048, 2176, 2304, 2432, 2560, 2688, 2816, 2944,
                      3072, 3200, 3328, 3456, 3584, 3712, 3840, 3968};
  out[c_input] = 0.0f;
  
  input[c_input] = A[T_offset];
  
  for (int k = 0; k < 4; k++) {
    int B_start = B_offset + (k<<12); // 32*64
    kernel[c_kernel] = B[B_start], kernel[c_kernel+1024] = B[B_start+1024];
    kernel[c_kernel+2048] = B[B_start+2048], kernel[c_kernel+3072] = B[B_start+3072];
    __syncthreads();
  
    float sum = 0;
    int y_tmp = (tY<<7)+(k<<5);
    for (int j = 0; j < 32; j++) {
      sum += input[y_tmp + j] * kernel[tX + B_stride[j]];
    }
    out[tY*128 + tX] += sum;
    __syncthreads();
  }
  
  C[T_offset] = out[c_input];
}



const char inputName256[] = "data/input_14_1_256.bin";
const char weight_winograd_Name256[] = "data/weight_winograd_256_256.bin";
const char bnBias_winograd_Name256[] = "data/bnBias_winograd_256.bin";
const char bnScale_winograd_Name256[] = "data/bnScale_winograd_256.bin";

#define d(input, i, j, Inz) ( input[Inz + i*768 + (j<<7)] )

extern "C" __global__ void kernel_256_winograd_BtdB(
  const float *__restrict__ pInputs,
        float *__restrict__ pOutputs)
{
  int Inx = blockIdx.x<<2, Iny0 = blockIdx.y<<2, Part = blockIdx.z,
      Iny1 = threadIdx.y, Inz = threadIdx.x;
  int Iny = Iny0+Iny1, stride_r = 4096, stride_c = 256; // 4096 = 16*256
  int c_glb_start = Inx*stride_r + Iny*stride_c + Inz + (Part<<7), c_input = Iny1*128 + Inz;

  __shared__ float input[6*6*128];

  int stride_768[6] = {0, 768, 1536, 2304, 3072, 3840}; // 768 = 6*128
  for (int i = 0; i < 6; i++) {
    input[c_input + stride_768[i]] = pInputs[c_glb_start + i*stride_r];
  }
  __syncthreads();

  float BTd[6];
  switch(Iny1) {
    case 0:
      for (int j = 0; j < 6; j++) {
        BTd[j] = d(input, 0, j, Inz)*4 - d(input, 2, j, Inz)*5 + d(input, 4, j, Inz);
      }
      break;
    case 1:
      for (int j = 0; j < 6; j++) {
        BTd[j] = -d(input, 1, j, Inz)*4 - d(input, 2, j, Inz)*4 + d(input, 3, j, Inz) + d(input, 4, j, Inz);
      }
      break;
    case 2:
      for (int j = 0; j < 6; j++) {
        BTd[j] = d(input, 1, j, Inz)*4 - d(input, 2, j, Inz)*4 - d(input, 3, j, Inz) + d(input, 4, j, Inz);
      }
      break;
    case 3:
      for (int j = 0; j < 6; j++) {
        BTd[j] = -d(input, 1, j, Inz)*2 - d(input, 2, j, Inz) + d(input, 3, j, Inz)*2 + d(input, 4, j, Inz);
      }
      break;
    case 4:
      for (int j = 0; j < 6; j++) {
        BTd[j] = d(input, 1, j, Inz)*2 - d(input, 2, j, Inz) - d(input, 3, j, Inz)*2 + d(input, 4, j, Inz);
      }
      break;
    case 5:
      for (int j = 0; j < 6; j++) {
        BTd[j] = d(input, 1, j, Inz)*4 - d(input, 3, j, Inz)*5 + d(input, 5, j, Inz);
      }
      break;
  }
  __syncthreads();

  int tmp_offset = Iny1*768+Inz;
  for (int i = 0; i < 6; i++) {
    input[tmp_offset + i*128] = BTd[i];
  }
  __syncthreads();

  float BTdB[6];
  switch(Iny1) {
    case 0:
      for (int i = 0; i < 6; i++) {
        BTdB[i] = 4*d(input, i, 0, Inz) - 5*d(input, i, 2, Inz) + d(input, i, 4, Inz);
      }
      break;
    case 1:
      for (int i = 0; i < 6; i++) {
        BTdB[i] = -4*d(input, i, 1, Inz) - 4*d(input, i, 2, Inz) + d(input, i, 3, Inz) + d(input, i, 4, Inz);
      }
      break;
    case 2:
      for (int i = 0; i < 6; i++) {
        BTdB[i] = 4*d(input, i, 1, Inz) - 4*d(input, i, 2, Inz) - d(input, i, 3, Inz) + d(input, i, 4, Inz);
      }
      break;
    case 3:
      for (int i = 0; i < 6; i++) {
        BTdB[i] = -2*d(input, i, 1, Inz) - d(input, i, 2, Inz) + 2*d(input, i, 3, Inz) + d(input, i, 4, Inz);
      }
      break;
    case 4:
      for (int i = 0; i < 6; i++) {
        BTdB[i] = 2*d(input, i, 1, Inz) - d(input, i, 2, Inz) - 2*d(input, i, 3, Inz) + d(input, i, 4, Inz);
      }
      break;
    case 5:
      for (int i = 0; i < 6; i++) {
        BTdB[i] = 4*d(input, i, 1, Inz) - 5*d(input, i, 3, Inz) + d(input, i, 5, Inz);
      }
      break;
  }
  __syncthreads();

  for (int i = 0; i < 6; i++) {
    pOutputs[(Iny1 + i*6)*4096 + (blockIdx.x*4+blockIdx.y)*256 + Inz + (Part<<7)] = BTdB[i];
  }
}

extern "C" __global__ void kernel_256_winograd_AtIA(
  const float *__restrict__ pInputs,
  const float *__restrict__ pBiases,
  const float *__restrict__ pScales,
        float *__restrict__ pOutputs)
{
  int Tilex = blockIdx.x, Inx = threadIdx.x;
  int Tiley = blockIdx.y, Iny = threadIdx.y;
  int kz = blockIdx.z; 
  int c_input = Inx*6 + Iny;

  __shared__ float bias, scale;
  __shared__ float input[6*6];

  input[c_input] = pInputs[c_input*16*256 + (Tilex*4+Tiley)*256 + kz];
  bias = pBiases[kz];
  scale = pScales[kz];
  __syncthreads();

  float tmp = 0;
  switch(Inx) {
    case 0:
      tmp = input[Iny] + input[6+Iny] + input[12+Iny] + input[18+Iny] + input[24+Iny];
      break;
    case 1:
      tmp = input[6+Iny] - input[12+Iny] + 2*input[18+Iny] - 2*input[24+Iny];
      break;
    case 2:
      tmp = input[6+Iny] + input[12+Iny] + 4*input[18+Iny] + 4*input[24+Iny];
      break;
    case 3:
      tmp = input[6+Iny] - input[12+Iny] + 8*input[18+Iny] - 8*input[24+Iny] + input[30+Iny];
      break;
  }
  __syncthreads();

  input[c_input] = tmp;
  __syncthreads();

  if (Inx > 3 || (Tilex == 3 && Inx > 1)) return;

  int x;
  float o;
  switch(Iny) {
    case 0:
      x = Inx*6;
      o = scale*(input[x]+input[x+1]+input[x+2]+input[x+3]+input[x+4]) + bias;
      pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+1)*256 + kz] = o > 0 ? o : 0;
      break;
    case 1:
      x = Inx*6;
      o = scale*(input[x+1] - input[x+2] + 2*input[x+3] - 2*input[x+4]) + bias;
      pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+2)*256 + kz] = o > 0 ? o : 0;
      break;
    case 2:
      if (Tiley == 3) break;
      x = Inx*6;
      o = scale*(input[x+1] + input[x+2] + 4*input[x+3] + 4*input[x+4]) + bias;
      pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+3)*256 + kz] = o > 0 ? o : 0;
      break;
    case 3:
      if (Tiley == 3) break;
      x = Inx*6;
      o = scale*(input[x+1] - input[x+2] + 8*input[x+3] - 8*input[x+4] + input[x+5]) + bias;
      pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+4)*256 + kz] = o > 0 ? o : 0;
      break;
  }
}

extern "C" __global__ void kernel_256_OuterProduct_256(
  const float *__restrict__ A,
  const float *__restrict__ B,
        float *__restrict__ C)
{
  int Tile = blockIdx.x, Part = blockIdx.y, 
      tX = threadIdx.x, tY = threadIdx.y;
  int c_input = tY*256 + tX,
      c_kernel = c_input,
      T_offset = (Tile<<12) + (Part<<11) + c_input, B_offset = (Tile<<16) + c_kernel;

  __shared__ float input[8*256 + 32*256 + 8*256];
  float *kernel = input + 2048, *out = kernel + 8192;
  int B_stride[32] = {0, 256, 512, 768, 1024, 1280, 1536, 1792, 
                      2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840,
                      4096, 4352, 4608, 4864, 5120, 5376, 5632, 5888,
                      6144, 6400, 6656, 6912, 7168, 7424, 7680, 7936};
  out[c_input] = 0.0f;
  out[c_input+1024] = 0;

  input[c_input] = A[T_offset];
  input[c_input+1024] = A[T_offset+1024];

  for (int k = 0; k < 8; k++) {
    int B_start = B_offset + (k<<13); // 32*64
    kernel[c_kernel] = B[B_start], kernel[c_kernel+1024] = B[B_start+1024];
    kernel[c_kernel+2048] = B[B_start+2048], kernel[c_kernel+3072] = B[B_start+3072];
    kernel[c_kernel+4096] = B[B_start+4096], kernel[c_kernel+5120] = B[B_start+5120];
    kernel[c_kernel+6144] = B[B_start+6144], kernel[c_kernel+7168] = B[B_start+7168];

    __syncthreads();

    float sum = 0, sum1 = 0;
    int y_tmp = (tY<<8)+(k<<5), y_tmp1 = y_tmp+1024;
    for (int j = 0; j < 32; j++) {
      sum += input[y_tmp + j] * kernel[tX + B_stride[j]];
      sum1 += input[y_tmp1 + j] * kernel[tX + B_stride[j]];
    }
    out[c_input] += sum;
    out[c_input+1024] += sum1;
    __syncthreads();
  }

  C[T_offset] = out[c_input];
  C[T_offset+1024] = out[c_input+1024];
}


