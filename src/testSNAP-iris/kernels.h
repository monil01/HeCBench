extern "C" __global__ void reset_ulisttot(COMPLEX *ulisttot, const int ulisttot_size) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ulisttot_size) ulisttot[i] = {0.0, 0.0};
}

extern "C" __global__ void set_ulisttot(
    COMPLEX *__restrict__ ulisttot,
    const int*__restrict__ idxu_block, 
    const int num_atoms,
    const int twojmax,
    const double wself) 
{
  int natom = blockIdx.x * blockDim.x + threadIdx.x;
  if (natom < num_atoms) 
    for (int j = 0; j <= twojmax; j++) {
      int jju = idxu_block[j];
      for (int ma = 0; ma <= j; ma++) {
        ulisttot[INDEX_2D(natom, jju)] = { wself, 0.0 };
        jju += j + 2;
      }
    }
}

extern "C" __global__ void update_ulisttot(
    const double*__restrict__ rij, 
    const double*__restrict__ rcutij,
    const double*__restrict__ wj, 
    const int*__restrict__ ulist_parity, 
    const int*__restrict__ idxu_block, 
    const double*__restrict__ rootpqarray, 
    COMPLEX *__restrict__ ulist, 
    COMPLEX *__restrict__ ulisttot, 
    const int num_atoms,
    const int num_nbor,
    const int switch_flag, 
    const int twojmax, 
    const int jdimpq)
{

  int natom = blockIdx.x * blockDim.x + threadIdx.x;
  int nbor = blockIdx.y * blockDim.y + threadIdx.y;

  if (natom < num_atoms && nbor < num_nbor) {
    double x = rij[ULIST_INDEX(natom, nbor, 0)];
    double y = rij[ULIST_INDEX(natom, nbor, 1)];
    double z = rij[ULIST_INDEX(natom, nbor, 2)];
    double rsq = x * x + y * y + z * z;
    double r = sqrt(rsq);

    double theta0 = (r - rmin0) * rfac0 * MY_PI / (rcutij[INDEX_2D(natom, nbor)] - rmin0);
    double z0 = r / tan(theta0);

    double rootpq;
    int jju, jjup;

    // compute Cayley-Klein parameters for unit quaternion

    double r0inv = 1.0 / sqrt(r * r + z0 * z0);
    double a_r = r0inv * z0;
    double a_i = -r0inv * z;
    double b_r = r0inv * y;
    double b_i = -r0inv * x;

    double sfac;

    sfac = compute_sfac(r, rcutij[INDEX_2D(natom, nbor)], switch_flag);
    sfac *= wj[INDEX_2D(natom, nbor)];

    // Recursion relations
    // VMK Section 4.8.2

    //   u[j,ma,mb] = Sqrt((j-ma)/(j-mb)) a* u[j-1,ma,mb]
    //               -Sqrt((ma)/(j-mb)) b* u[j-1,ma-1,mb]

    //   u[j,ma,mb] = Sqrt((j-ma)/(mb)) b u[j-1,ma,mb-1]
    //                Sqrt((ma)/(mb)) a u[j-1,ma-1,mb-1]

    // initialize first entry
    // initialize top row of each layer to zero
    ulist[ULIST_INDEX(natom, nbor, 0)].re = 1.0;
    ulist[ULIST_INDEX(natom, nbor, 0)].im = 0.0;

    // skip over right half of each uarray
    jju = 1;
    for (int j = 1; j <= twojmax; j++) {
      int deljju = j + 1;
      for (int mb = 0; 2 * mb <= j; mb++) {
        ulist[ULIST_INDEX(natom, nbor, jju)].re = 0.0;
        ulist[ULIST_INDEX(natom, nbor, jju)].im = 0.0;
        jju += deljju;
      }
      int ncolhalf = deljju / 2;
      jju += deljju * ncolhalf;
    }

    jju = 1;
    jjup = 0;
    for (int j = 1; j <= twojmax; j++) {
      int deljju = j + 1;
      int deljjup = j;
      int mb_max = (j + 1) / 2;
      int ma_max = j;
      int m_max = ma_max * mb_max;

      // fill in left side of matrix layer from previous layer
      for (int m_iter = 0; m_iter < m_max; ++m_iter) {
        int mb = m_iter / ma_max;
        int ma = m_iter % ma_max;
        double up_r = ulist[ULIST_INDEX(natom, nbor, jjup)].re;
        double up_i = ulist[ULIST_INDEX(natom, nbor, jjup)].im;

        rootpq = rootpqarray[ROOTPQ_INDEX(j - ma, j - mb)];
        ulist[ULIST_INDEX(natom, nbor, jju)].re += rootpq * (a_r * up_r + a_i * up_i);
        ulist[ULIST_INDEX(natom, nbor, jju)].im += rootpq * (a_r * up_i - a_i * up_r);

        rootpq = rootpqarray[ROOTPQ_INDEX(ma + 1, j - mb)];
        ulist[ULIST_INDEX(natom, nbor, jju+1)].re = -rootpq * (b_r * up_r + b_i * up_i);
        ulist[ULIST_INDEX(natom, nbor, jju+1)].im = -rootpq * (b_r * up_i - b_i * up_r);

        // assign middle column i.e. mb+1

        if (2 * (mb + 1) == j) {
          rootpq = rootpqarray[ROOTPQ_INDEX(j - ma, mb + 1)];
          ulist[ULIST_INDEX(natom, nbor, jju+deljju)].re += rootpq * (b_r * up_r - b_i * up_i);
          ulist[ULIST_INDEX(natom, nbor, jju+deljju)].im += rootpq * (b_r * up_i + b_i * up_r);

          rootpq = rootpqarray[ROOTPQ_INDEX(ma + 1, mb + 1)];
          ulist[ULIST_INDEX(natom, nbor, jju+deljju+1)].re = rootpq * (a_r * up_r - a_i * up_i);
          ulist[ULIST_INDEX(natom, nbor, jju+deljju+1)].im = rootpq * (a_r * up_i + a_i * up_r);
        }

        jju++;
        jjup++;

        if (ma == ma_max - 1)
          jju++;
      }

      // copy left side to right side with inversion symmetry VMK 4.4(2)
      // u[ma-j][mb-j] = (-1)^(ma-mb)*Conj([u[ma][mb])
      // dependence on idxu_block could be removed
      // renamed counters b/c can not modify jju, jjup
      int jjui = idxu_block[j];
      int jjuip = jjui + (j + 1) * (j + 1) - 1;
      for (int mb = 0; 2 * mb < j; mb++) {
        for (int ma = 0; ma <= j; ma++) {
          ulist[ULIST_INDEX(natom, nbor, jjuip)].re = ulist_parity[jjui] * ulist[ULIST_INDEX(natom, nbor, jjui)].re;
          ulist[ULIST_INDEX(natom, nbor, jjuip)].im = ulist_parity[jjui] * -ulist[ULIST_INDEX(natom, nbor, jjui)].im;
          jjui++;
          jjuip--;
        }
      }

      // skip middle and right half cols
      // b/c no longer using idxu_block
      if (j % 2 == 0)
        jju += deljju;
      int ncolhalf = deljju / 2;
      jju += deljju * ncolhalf;
      int ncolhalfp = deljjup / 2;
      jjup += deljjup * ncolhalfp;
    }

    sfac = compute_sfac(r, rcutij[INDEX_2D(natom, nbor)], switch_flag);
    sfac *= wj[INDEX_2D(natom, nbor)];

    for (int j = 0; j <= twojmax; j++) {
      int jju = idxu_block[j];
      for (int mb = 0; mb <= j; mb++)
        for (int ma = 0; ma <= j; ma++) {
          atomicAdd(&(ulisttot[INDEX_2D(natom, jju)].re), sfac * ulist[ULIST_INDEX(natom, nbor, jju)].re);
          atomicAdd(&(ulisttot[INDEX_2D(natom, jju)].im), sfac * ulist[ULIST_INDEX(natom, nbor, jju)].im);
          jju++;
        }
    }
  }
}

extern "C" __global__ void reset_ylist(COMPLEX *ylist, const int ylist_size) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ylist_size) ylist[i] = {0.0, 0.0};
}

extern "C" __global__ void compute_yi (
    const int*__restrict__ idxz,
    const double*__restrict__ idxzbeta,
    const double*__restrict__ cglist,
    const int*__restrict__ idxcg_block,
    const int*__restrict__ idxu_block,
    const int*__restrict__ idxdu_block,
    const COMPLEX*__restrict__ ulisttot,
          COMPLEX*__restrict__ ylist,
    const int num_atoms,
    const int idxz_max,
    const int jdim) 
{
  int natom = blockIdx.x * blockDim.x + threadIdx.x;
  int jjz = blockIdx.y * blockDim.y + threadIdx.y;
  if (jjz < idxz_max && natom < num_atoms) {
    const int j1 = idxz[IDXZ_INDEX(jjz, 0)];
    const int j2 = idxz[IDXZ_INDEX(jjz, 1)];
    const int j = idxz[IDXZ_INDEX(jjz, 2)];
    const int ma1min = idxz[IDXZ_INDEX(jjz, 3)];
    const int ma2max = idxz[IDXZ_INDEX(jjz, 4)];
    const int na = idxz[IDXZ_INDEX(jjz, 5)];
    const int mb1min = idxz[IDXZ_INDEX(jjz, 6)];
    const int mb2max = idxz[IDXZ_INDEX(jjz, 7)];
    const int nb = idxz[IDXZ_INDEX(jjz, 8)];

    const double betaj = idxzbeta[jjz];

    const double* cgblock = cglist + idxcg_block[j1 + jdim*j2 + jdim*jdim*j];

    int mb = (2 * (mb1min + mb2max) - j1 - j2 + j) / 2;
    int ma = (2 * (ma1min + ma2max) - j1 - j2 + j) / 2;
    const int jjdu = idxdu_block[j] + (j + 1) * mb + ma;

    int jju1 = idxu_block[j1] + (j1 + 1) * mb1min;
    int jju2 = idxu_block[j2] + (j2 + 1) * mb2max;
    int icgb = mb1min * (j2 + 1) + mb2max;

    double ztmp_r = 0.0;
    double ztmp_i = 0.0;

    // loop over columns of u1 and corresponding
    // columns of u2 satisfying Clebsch-Gordan constraint
    //      2*mb-j = 2*mb1-j1 + 2*mb2-j2

    for (int ib = 0; ib < nb; ib++) {

      double suma1_r = 0.0;
      double suma1_i = 0.0;

      int ma1 = ma1min;
      int ma2 = ma2max;
      int icga = ma1min * (j2 + 1) + ma2max;

      // loop over elements of row u1[mb1] and corresponding elements
      // of row u2[mb2] satisfying Clebsch-Gordan constraint
      //      2*ma-j = 2*ma1-j1 + 2*ma2-j2

      for (int ia = 0; ia < na; ia++) {
        suma1_r += cgblock[icga] *
          (ulisttot[INDEX_2D(natom, jju1 + ma1)].re * ulisttot[INDEX_2D(natom, jju2 + ma2)].re -
           ulisttot[INDEX_2D(natom, jju1 + ma1)].im * ulisttot[INDEX_2D(natom, jju2 + ma2)].im);

        suma1_i += cgblock[icga] *
          (ulisttot[INDEX_2D(natom, jju1 + ma1)].re * ulisttot[INDEX_2D(natom, jju2 + ma2)].im +
           ulisttot[INDEX_2D(natom, jju1 + ma1)].im * ulisttot[INDEX_2D(natom, jju2 + ma2)].re);

        ma1++;
        ma2--;
        icga += j2;
      } // end loop over ia

      ztmp_r += cgblock[icgb] * suma1_r;
      ztmp_i += cgblock[icgb] * suma1_i;
      jju1 += j1 + 1;
      jju2 -= j2 + 1;
      icgb += j2;
    } // end loop over ib

    // apply z(j1,j2,j,ma,mb) to unique element of y(j)

    atomicAdd(&(ylist[INDEX_2D(natom, jjdu)].re), betaj * ztmp_r);
    atomicAdd(&(ylist[INDEX_2D(natom, jjdu)].im), betaj * ztmp_i);

  } // end jjz and natom loop
}

extern "C" __global__ void compute_duidrj (
    const double *__restrict__ wj,
    const double *__restrict__ rij,
    const double *__restrict__ rcutij,
    const double*__restrict__ rootpqarray,
    const COMPLEX*__restrict__ ulist,
          COMPLEX*__restrict__ dulist,
    const int num_atoms,
    const int num_nbor,
    const int twojmax,
    const int idxdu_max,
    const int jdimpq,
    const int switch_flag)
{
  int natom = blockIdx.x * blockDim.x + threadIdx.x;
  int nbor = blockIdx.y * blockDim.y + threadIdx.y;
  if (natom < num_atoms && nbor < num_nbor) {
    double wj_in = wj[INDEX_2D(natom, nbor)];
    double rcut = rcutij[INDEX_2D(natom, nbor)];

    double x = rij[ULIST_INDEX(natom, nbor, 0)];
    double y = rij[ULIST_INDEX(natom, nbor, 1)];
    double z = rij[ULIST_INDEX(natom, nbor, 2)];
    double rsq = x * x + y * y + z * z;
    double r = sqrt(rsq);
    double rscale0 = rfac0 * MY_PI / (rcut - rmin0);
    double theta0 = (r - rmin0) * rscale0;
    double cs = cos(theta0);
    double sn = sin(theta0);
    double z0 = r * cs / sn;
    double dz0dr = z0 / r - (r * rscale0) * (rsq + z0 * z0) / rsq;

    compute_duarray(natom, nbor, num_atoms, num_nbor, 
        twojmax, idxdu_max, jdimpq, switch_flag,
        x, y, z, z0, r, dz0dr, wj_in, rcut,
        rootpqarray,
        ulist,
        dulist);
  }
}

extern "C" __global__ void compute_deidrj(
    const int*__restrict__ idxdu_block,
    const COMPLEX*__restrict__ dulist,
    const COMPLEX*__restrict__ ylist,
    double*__restrict__ dedr,
    const int num_atoms,
    const int num_nbor,
    const int twojmax,
    const int idxdu_max)
{
  int natom = blockIdx.x * blockDim.x + threadIdx.x;
  int nbor = blockIdx.y * blockDim.y + threadIdx.y;
  if (natom < num_atoms && nbor < num_nbor) {
    for (int k = 0; k < 3; k++)
      dedr[ULIST_INDEX(natom, nbor, k)] = 0.0;

    for (int j = 0; j <= twojmax; j++) {
      int jjdu = idxdu_block[j];

      for (int mb = 0; 2 * mb < j; mb++)
        for (int ma = 0; ma <= j; ma++) {

          double jjjmambyarray_r = ylist[INDEX_2D(natom, jjdu)].re;
          double jjjmambyarray_i = ylist[INDEX_2D(natom, jjdu)].im;

          for (int k = 0; k < 3; k++)
            dedr[ULIST_INDEX(natom, nbor, k)] +=
              dulist[DULIST_INDEX(natom, nbor, jjdu, k)].re * jjjmambyarray_r +
              dulist[DULIST_INDEX(natom, nbor, jjdu, k)].im * jjjmambyarray_i;
          jjdu++;
        } // end loop over ma mb

      // For j even, handle middle column

      if (j % 2 == 0) {

        int mb = j / 2;
        for (int ma = 0; ma < mb; ma++) {
          double jjjmambyarray_r = ylist[INDEX_2D(natom, jjdu)].re;
          double jjjmambyarray_i = ylist[INDEX_2D(natom, jjdu)].im;

          for (int k = 0; k < 3; k++)
            dedr[ULIST_INDEX(natom, nbor, k)] +=
              dulist[DULIST_INDEX(natom, nbor, jjdu, k)].re * jjjmambyarray_r +
              dulist[DULIST_INDEX(natom, nbor, jjdu, k)].im * jjjmambyarray_i;
          jjdu++;
        }

        double jjjmambyarray_r = ylist[INDEX_2D(natom, jjdu)].re;
        double jjjmambyarray_i = ylist[INDEX_2D(natom, jjdu)].im;

        for (int k = 0; k < 3; k++)
          dedr[ULIST_INDEX(natom, nbor, k)] +=
            (dulist[DULIST_INDEX(natom, nbor, jjdu, k)].re * jjjmambyarray_r +
             dulist[DULIST_INDEX(natom, nbor, jjdu, k)].im * jjjmambyarray_i) *
            0.5;
        jjdu++;

      } // end if jeven

    } // end loop over j

    for (int k = 0; k < 3; k++)
      dedr[ULIST_INDEX(natom, nbor, k)] *= 2.0;
  }
}
