/** GPU Laplace solver using optimized red-black Gauss Seidel with SOR solver
 *
 * author Kyle E. Niemeyer
 * date 09/21/2012
 *
 * Solves Laplace's equation in 2D (e.g., heat conduction in a rectangular plate)
 * on GPU using CUDA with the red-black Gauss Seidel with sucessive overrelaxation
 * (SOR) that has been "optimized". This means that the red and black kernels
 * only loop over their respective cells, instead of over all cells and skipping
 * even/odd cells. This requires separate arrays for red and black cells.
 *
 * Boundary conditions:
 * T = 0 at x = 0, x = L, y = 0
 * T = TN at y = H
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iris/iris.hpp>
#include <stdexcept>
static void check(int rc){if(rc!=IRIS_SUCCESS)throw std::runtime_error("IRIS API failure");}
template<class T> static void* arg(const T& value){return const_cast<T*>(&value);}
static int hip_device(){
  int n=0;check(iris_device_count(&n));
  for(int d=0;d<n;++d){int backend=0,type=0;char name[256]={};
    check(iris_device_info(d,iris_backend,&backend,nullptr));check(iris_device_info(d,iris_type,&type,nullptr));
    check(iris_device_info(d,iris_name,name,nullptr));
    printf("IRIS device %d: %s backend=%d type=%d\n",d,name,backend,type);
    if(backend==iris_hip&&(type&iris_gpu))return d;
  }
  throw std::runtime_error("no IRIS HIP GPU");
}

#include <vector>
#include "timer.h"
#include "config.h"
#include "reference.h"


/** Function to evaluate coefficient matrix and right-hand side vector.
 *
 * \param[in]   rowmax   number of rows
 * \param[in]   colmax   number of columns
 * \param[in]   th_cond  thermal conductivity
 * \param[in]   dx       grid size in x dimension (uniform)
 * \param[in]   dy       grid size in y dimension (uniform)
 * \param[in]   width    width of plate (z dimension)
 * \param[in]   TN       temperature at top boundary
 * \param[out]  aP       array of self coefficients
 * \param[out]  aW       array of west neighbor coefficients
 * \param[out]  aE       array of east neighbor coefficients
 * \param[out]  aS       array of south neighbor coefficients
 * \param[out]  aN       array of north neighbor coefficients
 * \param[out]  b        right-hand side array
 */
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


/** Main function that solves Laplace's equation in 2D (heat conduction in plate)
 *
 * Contains iteration loop for red-black Gauss-Seidel with SOR GPU kernels
 */
int main (int argc,char** argv) {

  // size of plate
  Real L = 1.0;
  Real H = 1.0;
  Real width = 0.01;

  // thermal conductivity
  Real th_cond = 1.0;

  // temperature at top boundary
  Real TN = 1.0;

  // SOR iteration tolerance
  Real tol = 1.e-6;

  // number of cells in x and y directions
  // including unused boundary cells
  int num_rows = (NUM / 2) + 2;
  int num_cols = NUM + 2;
  int size_temp = num_rows * num_cols;
  int size = NUM * NUM;

  // size of cells
  Real dx = L / NUM;
  Real dy = H / NUM;

  // iterations for Red-Black Gauss-Seidel with SOR
  int iter;
  int it_max = 1e6;

  // allocate memory
  Real *aP, *aW, *aE, *aS, *aN, *b;
  Real *temp_red, *temp_black;
  Real *temp_red_ref, *temp_black_ref;

  // arrays of coefficients
  aP = (Real *) calloc (size, sizeof(Real));
  aW = (Real *) calloc (size, sizeof(Real));
  aE = (Real *) calloc (size, sizeof(Real));
  aS = (Real *) calloc (size, sizeof(Real));
  aN = (Real *) calloc (size, sizeof(Real));

  // RHS
  b = (Real *) calloc (size, sizeof(Real));

  // temperature arrays
  temp_red = (Real *) calloc (size_temp, sizeof(Real));
  temp_black = (Real *) calloc (size_temp, sizeof(Real));
  temp_red_ref = (Real *) calloc (size_temp, sizeof(Real));
  temp_black_ref = (Real *) calloc (size_temp, sizeof(Real));

  // set coefficients
  fill_coeffs (NUM, NUM, th_cond, dx, dy, width, TN, aP, aW, aE, aS, aN, b);

  int i;
  for (i = 0; i < size_temp; ++i) {
    temp_red[i] = ZERO;
    temp_black[i] = ZERO;
  }

  // block and grid dimensions
  size_t gws[]={size_t((NUM/2+BLOCK_SIZE-1)/BLOCK_SIZE)*BLOCK_SIZE,size_t(NUM)},lws[]={BLOCK_SIZE,2};

  // print problem info
  printf("Problem size: %d x %d \n", NUM, NUM);

  iris::Platform platform;check(platform.init(&argc,&argv,true));int device=hip_device();
  iris::DMem aP_d(aP,size*sizeof(Real));
  iris::DMem aW_d(aW,size*sizeof(Real));
  iris::DMem aE_d(aE,size*sizeof(Real));
  iris::DMem aS_d(aS,size*sizeof(Real));
  iris::DMem aN_d(aN,size*sizeof(Real));
  iris::DMem b_d(b,size*sizeof(Real));
  iris::DMem temp_red_d(temp_red,size_temp*sizeof(Real));
  iris::DMem temp_black_d(temp_black,size_temp*sizeof(Real));
  size_t size_norm=size_temp,parts=(size_norm+255)/256;
  std::vector<Real> residual(size_norm,ZERO),partial(parts);
  Real norm_scalar=ZERO;
  iris::DMem bl_norm_L2_d(residual.data(),size_norm*sizeof(Real)),partial_d(partial.data(),parts*sizeof(Real));
  iris::DMem norm_L2_d(&norm_scalar,sizeof(Real));
  size_t reduction_gws=parts*256,reduction_lws=256,finish_gws=256;
  auto reduce=[&](){
    iris::Task ptask;void* pa[]={&bl_norm_L2_d,&partial_d,&size_norm};int pi[]={iris_r,iris_w,sizeof(size_norm)};
    check(ptask.kernel("partial_sum",1,nullptr,&reduction_gws,&reduction_lws,3,pa,pi));check(ptask.submit(device,nullptr,true));
    iris::Task ftask;void* fa[]={&partial_d,&norm_L2_d,&parts};int fi[]={iris_r,iris_w,sizeof(parts)};
    check(ftask.kernel("finish_sum",1,nullptr,&finish_gws,&reduction_lws,3,fa,fi));
    check(ftask.flush_out(norm_L2_d));check(ftask.submit(device,nullptr,true));return norm_scalar;
  };
  auto update=[&](const char* name,iris::DMem& input,iris::DMem& output){
    void* args[]={&aP_d,&aW_d,&aE_d,&aS_d,&aN_d,&b_d,&input,&output,&bl_norm_L2_d};
    int info[]={iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_rw,iris_rw};
    iris::Task task;check(task.kernel(name,2,nullptr,gws,lws,9,args,info));check(task.submit(device,nullptr,true));
  };
  StartTimer();

  // iteration loop
  for (iter = 1; iter <= it_max; ++iter) {

    Real norm_L2 = ZERO;

    update("red_kernel",temp_black_d,temp_red_d);
    Real red_norm=reduce();
    update("black_kernel",temp_red_d,temp_black_d);
    norm_L2=reduce()+red_norm;

    // calculate residual
    norm_L2 = sqrt(norm_L2 / ((Real)size));

    if (iter % 1000 == 0) printf("%5d, %0.6f\n", iter, norm_L2);

    // if tolerance has been reached, end SOR iterations
    if (norm_L2 < tol) break;
  }


  double runtime = GetTimer();
  printf("Total time for %i iterations: %f s\n", iter, runtime / 1000.0);

  // transfer final temperature values back
  iris::Task flush;check(flush.flush_out(temp_red_d));check(flush.flush_out(temp_black_d));check(flush.submit(device,nullptr,true));

  // Reference
  int count = 0;

  for (iter = 1; iter <= it_max; ++iter) {
    Real norm_L2;
    norm_L2 = red_ref(aP, aW, aE, aS, aN, b, temp_black_ref, temp_red_ref);
    norm_L2 += black_ref (aP, aW, aE, aS, aN, b, temp_red_ref, temp_black_ref);
    norm_L2 = sqrt(norm_L2 / ((Real)size));
    if (norm_L2 < tol) break;
  }

  // print temperature data to file
  FILE * pfile;
  pfile = fopen("temperature.dat", "w");

  if (pfile != NULL) {
    fprintf(pfile, "#x\ty\ttemp(K)\n");

    int row, col;
    for (row = 1; row < NUM + 1; ++row) {
      for (col = 1; col < NUM + 1; ++col) {
        Real x_pos = (col - 1) * dx + (dx / 2);
        Real y_pos = (row - 1) * dy + (dy / 2);

        if ((row + col) % 2 == 0) {
          // even, so red cell
          int ind = col * num_rows + (row + (col % 2)) / 2;
          if ((temp_red[ind] - temp_red_ref[ind]) >= 1e-3f) count++;
          fprintf(pfile, "%f\t%f\t%f\n", x_pos, y_pos, temp_red[ind]);
        } else {
          // odd, so black cell
          int ind = col * num_rows + (row + ((col + 1) % 2)) / 2;
          if ((temp_black[ind] - temp_black_ref[ind]) >= 1e-3f) count++;
          fprintf(pfile, "%f\t%f\t%f\n", x_pos, y_pos, temp_black[ind]);
        }
      }
      fprintf(pfile, "\n");
    }
  }

  fclose(pfile);
  printf("%s\n", count == 0 ? "PASS" : "FAIL");


  free(aP);
  free(aW);
  free(aE);
  free(aS);
  free(aN);
  free(b);
  free(temp_red);
  free(temp_black);
  free(temp_red_ref);
  free(temp_black_ref);

  check(platform.finalize());
  return count==0 ? 0 : 1;
}
