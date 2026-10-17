#include <cstdio>
#include <vector>
#define Real float
#define ZERO 0.0f
#define TWO 2.0f
void fill_coeffs (int rowmax, int colmax, Real th_cond, Real dx, Real dy,
    Real width, Real TN, Real * aP, Real * aW, Real * aE,
    Real * aS, Real * aN, Real * b)
{
  int col, row;
  for (col = 0; col < colmax; ++col) {
    for (row = 0; row < rowmax; ++row) {
      int ind = col * rowmax + row;

      b[ind] = ZERO;
      Real SP = ZERO;

      if (col == 0) {
        // left BC: temp = 0
        aW[ind] = ZERO;
        SP = -TWO * th_cond * width * dy / dx;
      } else {
        aW[ind] = th_cond * width * dy / dx;
      }

      if (col == colmax - 1) {
        // right BC: temp = 0
        aE[ind] = ZERO;
        SP = -TWO * th_cond * width * dy / dx;
      } else {
        aE[ind] = th_cond * width * dy / dx;
      }

      if (row == 0) {
        // bottom BC: temp = 0
        aS[ind] = ZERO;
        SP = -TWO * th_cond * width * dx / dy;
      } else {
        aS[ind] = th_cond * width * dx / dy;
      }

      if (row == rowmax - 1) {
        // top BC: temp = TN
        aN[ind] = ZERO;
        b[ind] = TWO * th_cond * width * dx * TN / dy;
        SP = -TWO * th_cond * width * dx / dy;
      } else {
        aN[ind] = th_cond * width * dx / dy;
      }

      aP[ind] = aW[ind] + aE[ind] + aS[ind] + aN[ind] - SP;
    } // end for row
  } // end for col
} // end fill_coeffs



int main(){for(int n:{1,4}){std::vector<float>ap(n*n),aw(n*n),ae(n*n),as(n*n),an(n*n),rhs(n*n);fill_coeffs(n,n,1.f,1.f/n,1.f/n,.01f,1.f,ap.data(),aw.data(),ae.data(),as.data(),an.data(),rhs.data());int k=n*n-1;float conductance=.01f;float expected_diag=n==1?8*conductance:6*conductance;printf("{\"n\":%d,\"native_corner_aP\":%.9g,\"finite_volume_corner_aP\":%.9g,\"corner_rhs\":%.9g,\"native_single_cell_T\":%.9g,\"independent_single_cell_T\":%.9g}\n",n,ap[k],expected_diag,rhs[k],n==1?rhs[k]/ap[k]:0.f,n==1?rhs[k]/expected_diag:0.f);}}
