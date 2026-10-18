#include <cmath>
#include <cstdio>
#include <algorithm>
using std::abs;
struct double4 {double x,y,z,w;};
static  inline void assemble_R(int m, double4 &sums, double *smem_svds)
{
  // Assemble R.

  double x0 = smem_svds[0];
  double x1 = smem_svds[1];
  double x2 = smem_svds[2];

  double x0_sq = x0 * x0;

  double sum1 = sums.x - x0;
  double sum2 = sums.y - x0_sq;
  double sum3 = sums.z - x0_sq*x0;
  double sum4 = sums.w - x0_sq*x0_sq;

  double m_as_dbl = (double) m;
  double sigma = m_as_dbl - 1.0;
  double mu = sqrt(m_as_dbl);
  double v0 = -sigma / (1.0 + mu);
  double v0_sq = v0*v0;
  double beta = 2.0 * v0_sq / (sigma + v0_sq);
  
  double inv_v0 = 1.0 / v0;
  double one_min_beta = 1.0 - beta;
  double beta_div_v0  = beta * inv_v0;
  
  smem_svds[0] = mu;
  smem_svds[1] = one_min_beta*x0 - beta_div_v0*sum1;
  smem_svds[2] = one_min_beta*x0_sq - beta_div_v0*sum2;
  
  // Rank update coefficients.
  
  double beta_div_v0_sq = beta_div_v0 * inv_v0;
  
  double c1 = beta_div_v0_sq*sum1 + beta_div_v0*x0;
  double c2 = beta_div_v0_sq*sum2 + beta_div_v0*x0_sq;

  // 2nd step of QR.
  
  double x1_sq = x1*x1;

  sum1 -= x1;
  sum2 -= x1_sq;
  sum3 -= x1_sq*x1;
  sum4 -= x1_sq*x1_sq;
  
  x0 = x1-c1;
  x0_sq = x0*x0;
  sigma = sum2 - 2.0*c1*sum1 + (m_as_dbl-2.0)*c1*c1;
  if( abs(sigma) < 1.0e-16 )
    beta = 0.0;
  else
  {
    mu = sqrt(x0_sq + sigma);
    if( x0 <= 0.0 )
      v0 = x0 - mu;
    else
      v0 = -sigma / (x0 + mu);
    v0_sq = v0*v0;
    beta = 2.0*v0_sq / (sigma + v0_sq);
  }
  
  inv_v0 = 1.0 / v0;
  beta_div_v0 = beta * inv_v0;
  
  // The coefficient to perform the rank update.
  double c3 = (sum3 - c1*sum2 - c2*sum1 + (m_as_dbl-2.0)*c1*c2)*beta_div_v0;
  double c4 = (x1_sq-c2)*beta_div_v0 + c3*inv_v0;
  double c5 = c1*c4 - c2;
  
  one_min_beta = 1.0 - beta;
  
  // Update R. 
  smem_svds[3] = one_min_beta*x0 - beta_div_v0*sigma;
  smem_svds[4] = one_min_beta*(x1_sq-c2) - c3;
  
  // 3rd step of QR.
  
  double x2_sq = x2*x2;

  sum1 -= x2;
  sum2 -= x2_sq;
  sum3 -= x2_sq*x2;
  sum4 -= x2_sq*x2_sq;
  
  x0 = x2_sq-c4*x2+c5;
  sigma = sum4 - 2.0*c4*sum3 + (c4*c4 + 2.0*c5)*sum2 - 2.0*c4*c5*sum1 + (m_as_dbl-3.0)*c5*c5;
  if( abs(sigma) < 1.0e-12 )
    beta = 0.0;
  else
  {
    mu = sqrt(x0*x0 + sigma);
    if( x0 <= 0.0 )
      v0 = x0 - mu;
    else
      v0 = -sigma / (x0 + mu);
    v0_sq = v0*v0;
    beta = 2.0*v0_sq / (sigma + v0_sq);
  }
  
  // Update R.
  smem_svds[5] = (1.0-beta)*x0 - (beta/v0)*sigma;
}

static  double off_diag_norm(double A01, double A02, double A12)
{
  return sqrt(2.0 * (A01*A01 + A02*A02 + A12*A12));
}

static  inline void swap(double &x, double &y)
{
  double t = x; x = y; y = t;
}

static  inline void svd_3x3(int m, double4 &sums, double *smem_svds)
{
  // Assemble the R matrix.
  assemble_R(m, sums, smem_svds);

  // The matrix R.
  double R00 = smem_svds[0];
  double R01 = smem_svds[1];
  double R02 = smem_svds[2];
  double R11 = smem_svds[3];
  double R12 = smem_svds[4];
  double R22 = smem_svds[5];

  // We compute the eigenvalues/eigenvectors of A = R^T R.
  
  double A00 = R00*R00;
  double A01 = R00*R01;
  double A02 = R00*R02;
  double A11 = R01*R01 + R11*R11;
  double A12 = R01*R02 + R11*R12;
  double A22 = R02*R02 + R12*R12 + R22*R22;
  
  // We keep track of V since A = Sigma^2 V. Each thread stores a row of V.
  
  double V00 = 1.0, V01 = 0.0, V02 = 0.0;
  double V10 = 0.0, V11 = 1.0, V12 = 0.0;
  double V20 = 0.0, V21 = 0.0, V22 = 1.0;
  
  // The Jacobi algorithm is iterative. We fix the max number of iter and the minimum tolerance.
  
  const int max_iters = 16;
  const double tolerance = 1.0e-12;
  
  // Iterate until we reach the max number of iters or the tolerance.
 
  for( int iter = 0 ; off_diag_norm(A01, A02, A12) >= tolerance && iter < max_iters ; ++iter )
  {
    double c, s, B00, B01, B02, B10, B11, B12, B20, B21, B22;
    
    // Compute the Jacobi matrix for p=0 and q=1.
    
    c = 1.0, s = 0.0;
    if( A01 != 0.0 )
    {
      double tau = (A11 - A00) / (2.0 * A01);
      double sgn = tau < 0.0 ? -1.0 : 1.0;
      double t   = sgn / (sgn*tau + sqrt(1.0 + tau*tau));
      
      c = 1.0 / sqrt(1.0 + t*t);
      s = t*c;
    }
    
    // Update A = J^T A J and V = V J.
    
    B00 = c*A00 - s*A01;
    B01 = s*A00 + c*A01;
    B10 = c*A01 - s*A11;
    B11 = s*A01 + c*A11;
    B02 = A02;
    
    A00 = c*B00 - s*B10;
    A01 = c*B01 - s*B11;
    A11 = s*B01 + c*B11;
    A02 = c*B02 - s*A12;
    A12 = s*B02 + c*A12;
    
    B00 = c*V00 - s*V01;
    V01 = s*V00 + c*V01;
    V00 = B00;
    
    B10 = c*V10 - s*V11;
    V11 = s*V10 + c*V11;
    V10 = B10;
    
    B20 = c*V20 - s*V21;
    V21 = s*V20 + c*V21;
    V20 = B20;
    
    // Compute the Jacobi matrix for p=0 and q=2.
    
    c = 1.0, s = 0.0;
    if( A02 != 0.0 )
    {
      double tau = (A22 - A00) / (2.0 * A02);
      double sgn = tau < 0.0 ? -1.0 : 1.0;
      double t   = sgn / (sgn*tau + sqrt(1.0 + tau*tau));
      
      c = 1.0 / sqrt(1.0 + t*t);
      s = t*c;
    }
    
    // Update A = J^T A J and V = V J.
    
    B00 = c*A00 - s*A02;
    B01 = c*A01 - s*A12;
    B02 = s*A00 + c*A02;
    B20 = c*A02 - s*A22;
    B22 = s*A02 + c*A22;
    
    A00 = c*B00 - s*B20;
    A12 = s*A01 + c*A12;
    A02 = c*B02 - s*B22;
    A22 = s*B02 + c*B22;
    A01 = B01;
    
    B00 = c*V00 - s*V02;
    V02 = s*V00 + c*V02;
    V00 = B00;
    
    B10 = c*V10 - s*V12;
    V12 = s*V10 + c*V12;
    V10 = B10;
    
    B20 = c*V20 - s*V22;
    V22 = s*V20 + c*V22;
    V20 = B20;
    
    // Compute the Jacobi matrix for p=1 and q=2.
    
    c = 1.0, s = 0.0;
    if( A12 != 0.0 )
    {
      double tau = (A22 - A11) / (2.0 * A12);
      double sgn = tau < 0.0 ? -1.0 : 1.0;
      double t   = sgn / (sgn*tau + sqrt(1.0 + tau*tau));
      
      c = 1.0 / sqrt(1.0 + t*t);
      s = t*c;
    }
    
    // Update A = J^T A J and V = V J.
    
    B02 = s*A01 + c*A02;
    B11 = c*A11 - s*A12;
    B12 = s*A11 + c*A12;
    B21 = c*A12 - s*A22;
    B22 = s*A12 + c*A22;
    
    A01 = c*A01 - s*A02;
    A02 = B02;
    A11 = c*B11 - s*B21;
    A12 = c*B12 - s*B22;
    A22 = s*B12 + c*B22;
    
    B01 = c*V01 - s*V02;
    V02 = s*V01 + c*V02;
    V01 = B01;
    
    B11 = c*V11 - s*V12;
    V12 = s*V11 + c*V12;
    V11 = B11;
    
    B21 = c*V21 - s*V22;
    V22 = s*V21 + c*V22;
    V21 = B21;
  }

  // Swap the columns to have S[0] >= S[1] >= S[2].
  if( A00 < A11 )
  {
    swap(A00, A11);
    swap(V00, V01);
    swap(V10, V11);
    swap(V20, V21);
  }
  if( A00 < A22 )
  {
    swap(A00, A22);
    swap(V00, V02);
    swap(V10, V12);
    swap(V20, V22);
  }
  if( A11 < A22 )
  {
    swap(A11, A22);
    swap(V01, V02);
    swap(V11, V12);
    swap(V21, V22);
  }

  //printf("timestep=%3d, svd0=%.8lf svd1=%.8lf svd2=%.8lf\n", blockIdx.x, sqrt(A00), sqrt(A11), sqrt(A22));
  
  // Invert the diagonal terms and compute V*S^-1.
  
  double inv_S0 = abs(A00) < 1.0e-12 ? 0.0 : 1.0 / A00;
  double inv_S1 = abs(A11) < 1.0e-12 ? 0.0 : 1.0 / A11;
  double inv_S2 = abs(A22) < 1.0e-12 ? 0.0 : 1.0 / A22;

  // printf("SVD: timestep=%3d %12.8lf %12.8lf %12.8lf\n", blockIdx.x, sqrt(A00), sqrt(A11), sqrt(A22));
  
  double U00 = V00 * inv_S0; 
  double U01 = V01 * inv_S1; 
  double U02 = V02 * inv_S2;
  double U10 = V10 * inv_S0; 
  double U11 = V11 * inv_S1; 
  double U12 = V12 * inv_S2;
  double U20 = V20 * inv_S0; 
  double U21 = V21 * inv_S1; 
  double U22 = V22 * inv_S2;
  
  // Compute V*S^-1*V^T*R^T.
  
#ifdef WITH_FULL_W_MATRIX
  double B00 = U00*V00 + U01*V01 + U02*V02;
  double B01 = U00*V10 + U01*V11 + U02*V12;
  double B02 = U00*V20 + U01*V21 + U02*V22;
  double B10 = U10*V00 + U11*V01 + U12*V02;
  double B11 = U10*V10 + U11*V11 + U12*V12;
  double B12 = U10*V20 + U11*V21 + U12*V22;
  double B20 = U20*V00 + U21*V01 + U22*V02;
  double B21 = U20*V10 + U21*V11 + U22*V12;
  double B22 = U20*V20 + U21*V21 + U22*V22;
  
  smem_svds[ 6] = B00*R00 + B01*R01 + B02*R02;
  smem_svds[ 7] =           B01*R11 + B02*R12;
  smem_svds[ 8] =                     B02*R22;
  smem_svds[ 9] = B10*R00 + B11*R01 + B12*R02;
  smem_svds[10] =           B11*R11 + B12*R12;
  smem_svds[11] =                     B12*R22;
  smem_svds[12] = B20*R00 + B21*R01 + B22*R02;
  smem_svds[13] =           B21*R11 + B22*R12;
  smem_svds[14] =                     B22*R22;
#else
  double B00 = U00*V00 + U01*V01 + U02*V02;
  double B01 = U00*V10 + U01*V11 + U02*V12;
  double B02 = U00*V20 + U01*V21 + U02*V22;
  double B11 = U10*V10 + U11*V11 + U12*V12;
  double B12 = U10*V20 + U11*V21 + U12*V22;
  double B22 = U20*V20 + U21*V21 + U22*V22;
  
  smem_svds[ 6] = B00*R00 + B01*R01 + B02*R02;
  smem_svds[ 7] =           B01*R11 + B02*R12;
  smem_svds[ 8] =                     B02*R22;
  smem_svds[ 9] =           B11*R11 + B12*R12;
  smem_svds[10] =                     B12*R22;
  smem_svds[11] =                     B22*R22;
#endif
}

int main(){double a[12]={1,2,3};double4 s{36,204,1296,8772};svd_3x3(8,s,a);for(double x:a)printf("%.17g\n",x);}
