extern "C" __global__ void
hotspot3d(
    const float*__restrict__ tIn, 
    const float*__restrict__ pIn, 
          float*__restrict__ tOut,
    const int numCols, 
    const int numRows, 
    const int layers,
    const float ce, 
    const float cw,
    const float cn, 
    const float cs,
    const float ct,
    const float cb,
    const float cc,
    const float stepDivCap)
{
  float amb_temp = 80.0;

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int c = i + j * numCols;
  int xy = numCols * numRows;

  int W = (i == 0)        ? c : c - 1;
  int E = (i == numCols-1)     ? c : c + 1;
  int N = (j == 0)        ? c : c - numCols;
  int S = (j == numRows-1)     ? c : c + numCols;

  float temp1, temp2, temp3;
  temp1 = temp2 = tIn[c];
  temp3 = tIn[c+xy];
  tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
    + cn * tIn[N] + cb * temp1 + ct * temp3 + stepDivCap * pIn[c] + ct * amb_temp;
  c += xy;
  W += xy;
  E += xy;
  N += xy;
  S += xy;

  for (int k = 1; k < layers-1; ++k) {
    temp1 = temp2;
    temp2 = temp3;
    temp3 = tIn[c+xy];
    tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
      + cn * tIn[N] + cb * temp1 + ct * temp3 + stepDivCap * pIn[c] + ct * amb_temp;
    c += xy;
    W += xy;
    E += xy;
    N += xy;
    S += xy;
  }
  temp1 = temp2;
  temp2 = temp3;
  tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
    + cn * tIn[N] + cb * temp1 + ct * temp3 + stepDivCap * pIn[c] + ct * amb_temp;
}
