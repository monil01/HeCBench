// Conditional study reference: accumulate all exterior-face conductances.
// Original main.cu is preserved; see source_boundary_witness.cpp.
void fill_coeffs_corrected (int rowmax, int colmax, Real th_cond, Real dx, Real dy,
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
        SP -= TWO * th_cond * width * dy / dx;
      } else {
        aW[ind] = th_cond * width * dy / dx;
      }

      if (col == colmax - 1) {
        // right BC: temp = 0
        aE[ind] = ZERO;
        SP -= TWO * th_cond * width * dy / dx;
      } else {
        aE[ind] = th_cond * width * dy / dx;
      }

      if (row == 0) {
        // bottom BC: temp = 0
        aS[ind] = ZERO;
        SP -= TWO * th_cond * width * dx / dy;
      } else {
        aS[ind] = th_cond * width * dx / dy;
      }

      if (row == rowmax - 1) {
        // top BC: temp = TN
        aN[ind] = ZERO;
        b[ind] = TWO * th_cond * width * dx * TN / dy;
        SP -= TWO * th_cond * width * dx / dy;
      } else {
        aN[ind] = th_cond * width * dx / dy;
      }

      aP[ind] = aW[ind] + aE[ind] + aS[ind] + aN[ind] - SP;
    } // end for row
  } // end for col
} // end fill_coeffs
